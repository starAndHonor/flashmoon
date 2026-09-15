///|
/// WebGPU flash-attention forward driver (JS target: browser).
///
/// Uploads q/k/v to GPU storage buffers, dispatches the online-softmax
/// compute shader (one thread per query row), and reads back the output.

///|
extern "js" fn now_ms() -> Double =
  #|() => performance.now()

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
fn extract2d(
  t : @nparray.NpArray,
  name : String,
) -> (FixedArray[Float], Int, Int) {
  let td = t.value()
  if td.dims.length() != 2 {
    abort("\{name}: expected 2D tensor")
  }
  (td.data, td.dims[0], td.dims[1])
}

///|
/// Create a storage buffer pre-filled with `data`.
fn upload(
  device : GPUDevice,
  data : FixedArray[Float],
  extra_usage : Int,
) -> GPUBuffer {
  let buf = device.create_buffer(
    data.length() * 4,
    USAGE_STORAGE | extra_usage,
    mapped_at_creation=true,
  )
  buf.set_float_array(data)
  buf.unmap()
  buf
}

///|
/// Key-split count for the split-K path: ~256 keys per workgroup,
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

///|
/// Flash attention forward on GPU: q [n,d], k [m,d], v [m,dv] → [n,dv].
/// Callback-style (MoonBit async intrinsics crash moonc v0.10.5 on js).
///
/// Fast path (d <= 64, dv <= 64): split-K — kernel 1 computes per-(row,
/// key-chunk) online-softmax partials (32 threads/workgroup), kernel 2
/// merges them. Fallback (64 < dv <= 128): single tiled kernel, one
/// thread per query row.
pub fn flash_attention_gpu(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  on_done : (@nparray.NpArray) -> Unit,
  log? : (String) -> Unit = fn(_) { () },
) -> Unit {
  let (qd, n, d) = extract2d(q, "q")
  let (kd, m, dk) = extract2d(k, "k")
  let (vd, vm, dv) = extract2d(v, "v")
  if d != dk || m != vm {
    abort("shape mismatch")
  }
  if dv > 128 {
    abort("dv > 128 not supported by the WGSL register array")
  }
  if (n + 63) / 64 > 65535 {
    abort("ceil(n/64) > 65535 exceeds the dispatch grid.y limit")
  }
  let split_k = d <= 64 && dv <= 64 && d % 4 == 0 && dv % 4 == 0
  let t_start = now_ms()
  gpu().request_adapter(fn(adapter) {
    adapter.request_device(fn(device) {
      log("adapter: \{adapter.info_string()}")
      let t_device = now_ms()
      log(
        "device maxComputeWorkgroupStorageSize=\{device.max_workgroup_storage()}",
      )
      device.on_uncaptured_error(fn(msg) { log("GPU error: \{msg}") })
      // Both kernels stage BC=16 K/V tiles: 16 * (d + dv) * 4 bytes.
      let smem_need = 16 * d * 4 + 16 * dv * 4
      if smem_need > device.max_workgroup_storage() {
        abort(
          "shader needs \{smem_need}B workgroup storage, device allows \{device.max_workgroup_storage()}B",
        )
      }
      // Upload inputs; output buffer is storage + copy source
      let qb = upload(device, qd, 0)
      let kb = upload(device, kd, 0)
      let vb = upload(device, vd, 0)
      let outb = device.create_buffer(
        n * dv * 4,
        USAGE_STORAGE | USAGE_COPY_SRC,
        mapped_at_creation=false,
      )
      let encoder = device.create_command_encoder()
      let pass = encoder.begin_compute_pass()
      if split_k {
        let splits = pick_splits(m)
        let partb = device.create_buffer(
          n * splits * (dv + 4) * 4,
          USAGE_STORAGE,
          mapped_at_creation=false,
        )
        let pp1 = device.create_pipeline_and_bind_group(
          device.create_shader_module(partial_shader(n, m, d, dv, splits)),
          "main",
          qb,
          kb,
          vb,
          partb,
        )
        let pp2 = device.create_pipeline_and_bind_group2(
          device.create_shader_module(combine_shader(n, dv, splits)),
          "main",
          partb,
          outb,
        )
        pass.set_pipeline(pp1.pipeline())
        pass.set_bind_group(0, pp1.bind_group())
        pass.dispatch_workgroups(splits, (n + 63) / 64, 1)
        pass.set_pipeline(pp2.pipeline())
        pass.set_bind_group(0, pp2.bind_group())
        pass.dispatch_workgroups((n * (dv / 4) + 255) / 256, 1, 1)
        log("split-K: splits=\{splits} workgroups1=\{splits * ((n + 63) / 64)}")
      } else {
        let pp = device.create_pipeline_and_bind_group(
          device.create_shader_module(attention_shader(n, m, d, dv)),
          "main",
          qb,
          kb,
          vb,
          outb,
        )
        pass.set_pipeline(pp.pipeline())
        pass.set_bind_group(0, pp.bind_group())
        pass.dispatch_workgroups((n + 63) / 64, 1, 1)
      }
      pass.end()
      let readb = device.create_buffer(
        n * dv * 4,
        USAGE_MAP_READ | USAGE_COPY_DST,
        mapped_at_creation=false,
      )
      encoder.copy_buffer_to_buffer(outb, 0, readb, 0, n * dv * 4)
      device.get_queue().submit([encoder.finish()])
      let t_submit = now_ms()
      log(
        "  stages: init=\{t_device - t_start}ms upload+pipeline+encode+submit=\{t_submit - t_device}ms",
      )
      readb.map_async(MAP_MODE_READ, fn(_) {
        let t_done = now_ms()
        log("  gpu wait (kernels+copy+map): \{t_done - t_submit}ms")
        let result = readb.get_float_array()
        readb.unmap()
        on_done(@tensor.Tensor::from_host(result, [n, dv]))
      })
    })
  })
}
