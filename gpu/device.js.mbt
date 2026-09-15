///|
/// GPU device context: one adapter/device/queue for the whole session,
/// plus a pipeline cache (Tint compilation is ~100ms per shader — never
/// recompile per dispatch).

///|
const USAGE_STORAGE : Int = 0x0080

///|
const USAGE_COPY_DST : Int = 0x0008

///|
const USAGE_COPY_SRC : Int = 0x0004

///|
const USAGE_MAP_READ : Int = 0x0001

///|
const MAP_MODE_READ : Int = 0x0001

///|
pub struct Gpu {
  device : GPUDevice
  queue : GPUQueue
  pipelines : Map[String, GPUComputePipeline]
}

///|
pub fn Gpu::init(
  cb : (Gpu) -> Unit,
  log? : (String) -> Unit = fn(_) { () },
) -> Unit {
  gpu().request_adapter(fn(adapter) {
    log("adapter: \{adapter.info_string()}")
    adapter.request_device(fn(device) {
      log(
        "device maxComputeWorkgroupStorageSize=\{device.max_workgroup_storage()}",
      )
      cb({ device, queue: device.get_queue(), pipelines: Map([]) })
    })
  })
}

///|
/// Upload f32 data into a new GPU-resident storage buffer.
pub fn Gpu::upload(self : Gpu, data : FixedArray[Float]) -> GPUBuffer {
  let buf = self.device.create_buffer(
    data.length() * 4,
    USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST,
    mapped_at_creation=true,
  )
  buf.set_float_array(data)
  buf.unmap()
  buf
}

///|
/// Allocate a zero-filled GPU-resident storage buffer of `n` floats.
pub fn Gpu::alloc(self : Gpu, n : Int) -> GPUBuffer {
  self.device.create_buffer(
    n * 4,
    USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
}

///|
/// Shader module + pipeline, cached by `key` (shader text is baked with
/// dimensions, so the key must include them).
fn Gpu::pipeline(
  self : Gpu,
  key : String,
  src : () -> String,
) -> GPUComputePipeline {
  match self.pipelines.get(key) {
    Some(p) => p
    None => {
      let p = self.device.create_pipeline(
        self.device.create_shader_module(src()),
        "main",
      )
      self.pipelines[key] = p
      p
    }
  }
}

///|
/// Encode + submit one compute pass for `pipeline` with `bufs` bound at
/// 0..n, dispatching (gx, gy, 1) workgroups.
fn Gpu::run_pass(
  self : Gpu,
  pipeline : GPUComputePipeline,
  bufs : FixedArray[GPUBuffer],
  gx : Int,
  gy : Int,
) -> Unit {
  let bg = self.device.make_bind_group(pipeline, bufs)
  let enc = self.device.create_command_encoder()
  let pass = enc.begin_compute_pass()
  pass.set_pipeline(pipeline)
  pass.set_bind_group(0, bg)
  pass.dispatch_workgroups(gx, gy, 1)
  pass.end()
  self.queue.submit([enc.finish()])
}

///|
/// `cb` fires when all submitted work has finished (GPU-side).
pub fn Gpu::sync(self : Gpu, cb : (Unit) -> Unit) -> Unit {
  self.queue.on_submitted_done(fn(_) { cb(()) })
}

///|
/// Copy `n` floats back to host; `cb` receives them.
pub fn Gpu::readback(
  self : Gpu,
  buf : GPUBuffer,
  n : Int,
  cb : (FixedArray[Float]) -> Unit,
) -> Unit {
  let readb = self.device.create_buffer(
    n * 4,
    USAGE_MAP_READ | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
  let enc = self.device.create_command_encoder()
  enc.copy_buffer_to_buffer(buf, 0, readb, 0, n * 4)
  self.queue.submit([enc.finish()])
  readb.map_async(MAP_MODE_READ, fn(_) {
    let r = readb.get_float_array()
    readb.unmap()
    cb(r)
  })
}

///|
/// y[out] = W[out, in] @ x[in] — row-major W, GPU-resident throughout.
/// Requires in % 4 == 0, in <= 3072 (shared x staging budget).
pub fn Gpu::matvec(
  self : Gpu,
  w : GPUBuffer,
  x : GPUBuffer,
  y : GPUBuffer,
  out : Int,
  inn : Int,
) -> Unit {
  if inn % 4 != 0 || inn > 3072 {
    abort("matvec: in must be a multiple of 4 and <= 3072, got \{inn}")
  }
  let p = self.pipeline("gemv:\{out}x\{inn}", fn() { gemv_shader(out, inn) })
  self.run_pass(p, [w, x, y], (out + 7) / 8, 1)
}

///|
/// Y[s, out] = X[s, in] @ W[out, in]^T — row-major, GPU-resident.
/// Requires in % 4 == 0.
pub fn Gpu::matmul(
  self : Gpu,
  x : GPUBuffer,
  w : GPUBuffer,
  y : GPUBuffer,
  s : Int,
  out : Int,
  inn : Int,
) -> Unit {
  if inn % 4 != 0 {
    abort("matmul: in must be a multiple of 4, got \{inn}")
  }
  let p = self.pipeline("gemm:\{s}x\{out}x\{inn}", fn() {
    matmul_shader(s, out, inn)
  })
  self.run_pass(p, [x, w, y], (out + 15) / 16, (s + 15) / 16)
}

///|
/// Flash attention forward on GPU-resident buffers: q [n,d], k [m,d],
/// v [m,dv] → out [n,dv]. See gpu/shader.mbt for the kernel structure.
pub fn Gpu::flash_attention(
  self : Gpu,
  qb : GPUBuffer,
  kb : GPUBuffer,
  vb : GPUBuffer,
  outb : GPUBuffer,
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
  log? : (String) -> Unit = fn(_) { () },
) -> Unit {
  let split_k = d <= 64 && dv <= 64 && d % 4 == 0 && dv % 4 == 0
  // Both kernel families stage BC=16 K/V tiles: 16 * (d + dv) * 4 bytes.
  let smem_need = 16 * d * 4 + 16 * dv * 4
  if smem_need > self.device.max_workgroup_storage() {
    abort(
      "shader needs \{smem_need}B workgroup storage, device allows \{self.device.max_workgroup_storage()}B",
    )
  }
  let enc = self.device.create_command_encoder()
  let pass = enc.begin_compute_pass()
  if split_k {
    let splits = pick_splits(m)
    let partb = self.alloc(n * splits * (dv + 4))
    let p1 = self.pipeline("fa1:\{n},\{m},\{d},\{dv},\{splits}", fn() {
      partial_shader(n, m, d, dv, splits)
    })
    let p2 = self.pipeline("fa2:\{n},\{dv},\{splits}", fn() {
      combine_shader(n, dv, splits)
    })
    let bg1 = self.device.make_bind_group(p1, [qb, kb, vb, partb])
    let bg2 = self.device.make_bind_group(p2, [partb, outb])
    pass.set_pipeline(p1)
    pass.set_bind_group(0, bg1)
    pass.dispatch_workgroups(splits, (n + 63) / 64, 1)
    pass.set_pipeline(p2)
    pass.set_bind_group(0, bg2)
    pass.dispatch_workgroups((n * (dv / 4) + 255) / 256, 1, 1)
    log("split-K: splits=\{splits} workgroups1=\{splits * ((n + 63) / 64)}")
  } else {
    if dv > 128 {
      abort("dv > 128 not supported by the WGSL register array")
    }
    if (n + 63) / 64 > 65535 {
      abort("ceil(n/64) > 65535 exceeds the dispatch grid.y limit")
    }
    let p = self.pipeline("fa0:\{n},\{m},\{d},\{dv}", fn() {
      attention_shader(n, m, d, dv)
    })
    let bg = self.device.make_bind_group(p, [qb, kb, vb, outb])
    pass.set_pipeline(p)
    pass.set_bind_group(0, bg)
    pass.dispatch_workgroups((n + 63) / 64, 1, 1)
  }
  pass.end()
  self.queue.submit([enc.finish()])
}

///|
/// Key-split count for the split-K path: ~128 keys per workgroup,
/// clamped to [1, 32]. Grid.y = ceil(n/64) must stay under 65535.
fn pick_splits(m : Int) -> Int {
  let s = (m + 127) / 128
  if s < 1 {
    1
  } else if s > 32 {
    32
  } else {
    s
  }
}
