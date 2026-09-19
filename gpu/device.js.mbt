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
  /// Some((enc, pass)) while batching: ops append to ONE open compute
  /// pass until Gpu::flush (per-pass overhead >> per-dispatch on Dawn).
  batch : Ref[(GPUCommandEncoder, GPUComputePassEncoder)?]
}

///|
pub fn Gpu::init(
  cb : (Gpu) -> Unit,
  log? : (String) -> Unit = fn(_) { () },
) -> Unit {
  gpu().request_adapter(fn(adapter) {
    log("adapter: \{adapter.info_string()} | \{adapter.limit_string()}")
    adapter.request_device(fn(device) {
      log(
        "device maxComputeWorkgroupStorageSize=\{device.max_workgroup_storage()}",
      )
      device.on_uncaptured_error(fn(msg) { log("GPU ERROR: \{msg}") })
      cb({
        device,
        queue: device.get_queue(),
        pipelines: Map([]),
        batch: { val: None },
      })
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
/// Begin batching: subsequent compute ops append to ONE open compute pass
/// until Gpu::flush submits the encoder. A decode token is ~250 ops; on
/// Dawn, per-pass overhead dominates per-dispatch, so all ops share a pass.
pub fn Gpu::begin(self : Gpu) -> Unit {
  let enc = self.device.create_command_encoder()
  self.batch.val = Some((enc, enc.begin_compute_pass()))
}

///|
/// Submit the batched pass+encoder; no-op outside a batch.
pub fn Gpu::flush(self : Gpu) -> Unit {
  match self.batch.val {
    Some((enc, pass)) => {
      pass.end()
      self.queue.submit([enc.finish()])
      self.batch.val = None
    }
    None => ()
  }
}

///|
/// Encode one dispatch of `pipeline` with `bufs` bound at 0..n. Batched:
/// appends to the open batch pass; otherwise one-shot pass + submit.
fn Gpu::run_pass(
  self : Gpu,
  pipeline : GPUComputePipeline,
  bufs : FixedArray[GPUBuffer],
  gx : Int,
  gy : Int,
) -> Unit {
  let bg = self.device.make_bind_group(pipeline, bufs)
  match self.batch.val {
    Some((_, pass)) => {
      pass.set_pipeline(pipeline)
      pass.set_bind_group(0, bg)
      pass.dispatch_workgroups(gx, gy, 1)
    }
    None => {
      let enc = self.device.create_command_encoder()
      let pass = enc.begin_compute_pass()
      pass.set_pipeline(pipeline)
      pass.set_bind_group(0, bg)
      pass.dispatch_workgroups(gx, gy, 1)
      pass.end()
      self.queue.submit([enc.finish()])
    }
  }
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
  self.run_pass(p, [w, x, y], (out + 15) / 16, 1)
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

///|
/// Upload a host typed array (any length — no JS-array element cap).
pub fn Gpu::upload_typed(self : Gpu, data : F32Buf) -> GPUBuffer {
  let buf = self.device.create_buffer(
    data.byte_len(),
    USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
  self.queue.write_buffer_f32(buf, 0, data)
  buf
}

///|
/// One u32 of kernel params (rope/attention position base).
pub fn Gpu::alloc_u32(self : Gpu) -> GPUBuffer {
  self.device.create_buffer(
    4,
    USAGE_STORAGE | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
}

///|
pub fn Gpu::set_u32(self : Gpu, buf : GPUBuffer, v : Int) -> Unit {
  self.queue.write_u32(buf, v)
}

///|
/// Block copy of `n` floats between storage buffers. Batched: the open
/// compute pass is closed, the copy recorded, and a fresh pass reopened.
pub fn Gpu::copy(
  self : Gpu,
  src : GPUBuffer,
  src_off : Int,
  dst : GPUBuffer,
  dst_off : Int,
  n : Int,
) -> Unit {
  match self.batch.val {
    Some((enc, pass)) => {
      pass.end()
      enc.copy_buffer_to_buffer(src, src_off * 4, dst, dst_off * 4, n * 4)
      self.batch.val = Some((enc, enc.begin_compute_pass()))
    }
    None => {
      let enc = self.device.create_command_encoder()
      enc.copy_buffer_to_buffer(src, src_off * 4, dst, dst_off * 4, n * 4)
      self.queue.submit([enc.finish()])
    }
  }
}

///|
/// RMSNorm over `rows` rows of length `n`; in-place safe (x == y).
pub fn Gpu::rmsnorm(
  self : Gpu,
  x : GPUBuffer,
  w : GPUBuffer,
  y : GPUBuffer,
  rows : Int,
  n : Int,
  eps : Float,
) -> Unit {
  if n % 4 != 0 {
    abort("rmsnorm: n must be a multiple of 4, got \{n}")
  }
  let p = self.pipeline("rms:\{n}:\{eps}", fn() { rmsnorm_shader(n, eps) })
  self.run_pass(p, [x, w, y], rows, 1)
}

///|
/// NeoX RoPE in place over x [rows, heads*2*half]; pos = pos0 + row.
pub fn Gpu::rope(
  self : Gpu,
  x : GPUBuffer,
  cost : GPUBuffer,
  sint : GPUBuffer,
  pos0 : GPUBuffer,
  rows : Int,
  heads : Int,
  half : Int,
) -> Unit {
  let p = self.pipeline("rope:\{rows},\{heads},\{half}", fn() {
    rope_shader(rows, heads, half)
  })
  self.run_pass(p, [x, cost, sint, pos0], (rows * heads * half + 255) / 256, 1)
}

///|
/// y = silu(g) * u elementwise over n floats.
pub fn Gpu::silu_mul(
  self : Gpu,
  g : GPUBuffer,
  u : GPUBuffer,
  y : GPUBuffer,
  n : Int,
) -> Unit {
  if n % 4 != 0 {
    abort("silu_mul: n must be a multiple of 4, got \{n}")
  }
  let p = self.pipeline("silu:\{n}", fn() { silu_mul_shader(n) })
  self.run_pass(p, [g, u, y], (n / 4 + 255) / 256, 1)
}

///|
/// x += y elementwise over n floats (residual).
pub fn Gpu::add(self : Gpu, x : GPUBuffer, y : GPUBuffer, n : Int) -> Unit {
  if n % 4 != 0 {
    abort("add: n must be a multiple of 4, got \{n}")
  }
  let p = self.pipeline("add:\{n}", fn() { add_shader(n) })
  self.run_pass(p, [x, y], (n / 4 + 255) / 256, 1)
}

///|
/// Flash prefill attention (tiled online softmax) over KV caches
/// [max_pos, hkv*hd]: q [rows, hq*hd] -> out [rows, hq*hd], causal with
/// row r attending keys 0..pos0+r (pos0 = params[0]). See shader.mbt.
pub fn Gpu::attn_prefill(
  self : Gpu,
  q : GPUBuffer,
  kc : GPUBuffer,
  vc : GPUBuffer,
  out : GPUBuffer,
  pos0 : GPUBuffer,
  rows : Int,
  hq : Int,
  hkv : Int,
  hd : Int,
) -> Unit {
  if hd > 128 {
    abort("attn_prefill: register-array accumulator caps hd at 128, got \{hd}")
  }
  if hq % hkv != 0 {
    abort("attn_prefill: hq % hkv != 0")
  }
  let smem = 8 * hd * 4 * 2 // BC=8 K + V tiles
  if smem > self.device.max_workgroup_storage() {
    abort(
      "attn_prefill needs \{smem}B workgroup storage, device allows \{self.device.max_workgroup_storage()}B",
    )
  }
  let scale = 1.0 / hd.to_double().sqrt()
  let p = self.pipeline("attnp:\{rows},\{hq},\{hkv},\{hd}", fn() {
    attn_prefill_shader(rows, hq, hkv, hd, scale)
  })
  self.run_pass(p, [q, kc, vc, out, pos0], (rows + 63) / 64, hq)
}

///|
/// KV split count for flash-decoding: 16 heads x 16 splits = 256
/// workgroups — enough to fill the GPU even at decode (rows = 1).
pub const ATTN_DECODE_SPLITS : Int = 16

///|
/// Floats of partial scratch that attn_decode needs: hq * splits * (hd+4)
/// (per (head, split) record: m, l, 2 pad, acc[hd]).
pub fn Gpu::attn_decode_part_len(hq : Int, hd : Int) -> Int {
  hq * ATTN_DECODE_SPLITS * (hd + 4)
}

///|
/// Flash-decoding attention (split-KV partials + combine), decode shape:
/// q [1, hq*hd] over KV caches, m = params[0] + 1 keys -> out [1, hq*hd].
/// `part` is scratch of attn_decode_part_len(hq, hd) floats. The two
/// dispatches share the caller's pass; in-pass dispatch ordering makes the
/// combine see the partials.
pub fn Gpu::attn_decode(
  self : Gpu,
  q : GPUBuffer,
  kc : GPUBuffer,
  vc : GPUBuffer,
  out : GPUBuffer,
  part : GPUBuffer,
  pos0 : GPUBuffer,
  hq : Int,
  hkv : Int,
  hd : Int,
  max_pos : Int,
) -> Unit {
  if hd != 128 {
    abort("attn_decode: only hd=128 (thread-per-dim) is implemented, got \{hd}")
  }
  if hq % hkv != 0 {
    abort("attn_decode: hq % hkv != 0")
  }
  let scale = 1.0 / hd.to_double().sqrt()
  let splits = ATTN_DECODE_SPLITS
  let p1 = self.pipeline("attnd:\{hq},\{hkv},\{hd},\{max_pos}", fn() {
    attn_decode_shader(hq, hkv, hd, max_pos, splits, scale)
  })
  self.run_pass(p1, [q, kc, vc, part, pos0], splits, hq)
  let p2 = self.pipeline("attnc:\{hq},\{hd}", fn() {
    attn_combine_shader(hd, splits)
  })
  self.run_pass(p2, [part, out], hq, 1)
}

///|
/// Upload raw bytes (e.g. a BF16 safetensors payload, zero conversion).
pub fn Gpu::upload_raw(
  self : Gpu,
  data : Bytes,
  off : Int,
  byte_len : Int,
) -> GPUBuffer {
  let buf = self.device.create_buffer(
    byte_len,
    USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
  self.queue.write_buffer_raw(buf, bytes_view(data, off, byte_len))
  buf
}

///|
/// Upload a raw typed array (e.g. host-converted bf16 test data).
pub fn Gpu::upload_raw_buf(self : Gpu, data : RawBuf) -> GPUBuffer {
  let buf = self.device.create_buffer(
    data.byte_len(),
    USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
  self.queue.write_buffer_raw(buf, data)
  buf
}

///|
/// Allocate an `n`-entry u32 buffer and fill it (token ids).
pub fn Gpu::upload_u32(self : Gpu, data : Array[Int]) -> GPUBuffer {
  let buf = self.device.create_buffer(
    data.length() * 4,
    USAGE_STORAGE | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
  self.queue.write_u32_array(buf, data)
  buf
}

///|
/// Embedding gather, bf16 weights: dst[row, :] = emb[ids[row], :] for
/// rows token rows of length n.
pub fn Gpu::embed_rows(
  self : Gpu,
  emb : GPUBuffer,
  ids : GPUBuffer,
  dst : GPUBuffer,
  rows : Int,
  n : Int,
) -> Unit {
  if n % 2 != 0 {
    abort("embed_rows: n must be even, got \{n}")
  }
  let p = self.pipeline("embed:\{rows},\{n}", fn() {
    embed_rows_shader(rows, n)
  })
  self.run_pass(p, [emb, ids, dst], (rows * (n / 2) + 255) / 256, 1)
}

///|
/// Fused QKV projection (decode): qout = Wq x, kout = Wk x, and
/// vc[params[0]] = Wv x (v written straight into the KV cache).
pub fn Gpu::qkv_fused(
  self : Gpu,
  x : GPUBuffer,
  wq : GPUBuffer,
  wk : GPUBuffer,
  wv : GPUBuffer,
  qout : GPUBuffer,
  kout : GPUBuffer,
  vc : GPUBuffer,
  params : GPUBuffer,
  nq : Int,
  nkv : Int,
  inn : Int,
) -> Unit {
  let p = self.pipeline("qkv:\{nq},\{nkv},\{inn}", fn() {
    qkv_fused_shader(nq, nkv, inn)
  })
  self.run_pass(
    p,
    [x, wq, wk, wv, qout, kout, vc, params],
    (nq + 2 * nkv + 7) / 8,
    1,
  )
}

///|
/// Fused gate+up projection (decode): gout = Wg x, uout = Wu x.
pub fn Gpu::gemv2_fused(
  self : Gpu,
  x : GPUBuffer,
  wg : GPUBuffer,
  wu : GPUBuffer,
  gout : GPUBuffer,
  uout : GPUBuffer,
  n : Int,
  inn : Int,
) -> Unit {
  let p = self.pipeline("gemv2:\{n},\{inn}", fn() { gemv2_fused_shader(n, inn) })
  self.run_pass(p, [x, wg, wu, gout, uout], (2 * n + 7) / 8, 1)
}

///|
/// Fused per-head RMSNorm + NeoX RoPE for q (in place) and k (written
/// into the KV cache at params[0]). Decode only (one token row).
pub fn Gpu::qknorm_rope(
  self : Gpu,
  q : GPUBuffer,
  k : GPUBuffer,
  qnw : GPUBuffer,
  knw : GPUBuffer,
  kc : GPUBuffer,
  cost : GPUBuffer,
  sint : GPUBuffer,
  params : GPUBuffer,
  hq : Int,
  hkv : Int,
  hd : Int,
  eps : Float,
) -> Unit {
  if hd != 128 {
    abort("qknorm_rope: only hd=128, got \{hd}")
  }
  let p = self.pipeline("qknr:\{hq},\{hkv},\{hd}", fn() {
    qknorm_rope_shader(hq, hkv, hd, eps)
  })
  self.run_pass(p, [q, k, qnw, knw, kc, cost, sint, params], hq + hkv, 1)
}

///|
/// Fused residual add + RMSNorm: x[r] += d[r]; y[r] = rmsnorm(x[r]) * w.
pub fn Gpu::add_rmsnorm(
  self : Gpu,
  x : GPUBuffer,
  d : GPUBuffer,
  w : GPUBuffer,
  y : GPUBuffer,
  rows : Int,
  n : Int,
  eps : Float,
) -> Unit {
  if n % 4 != 0 {
    abort("add_rmsnorm: n must be a multiple of 4, got \{n}")
  }
  let p = self.pipeline("arms:\{n}:\{eps}", fn() { add_rmsnorm_shader(n, eps) })
  self.run_pass(p, [x, d, w, y], rows, 1)
}

///|
/// Down-projection GEMV with fused silu(g)*u activation on staging.
pub fn Gpu::matvec_silu(
  self : Gpu,
  w : GPUBuffer,
  g : GPUBuffer,
  u : GPUBuffer,
  y : GPUBuffer,
  out : Int,
  inn : Int,
) -> Unit {
  let p = self.pipeline("gemvs:\{out}x\{inn}", fn() {
    gemv_silu_shader(out, inn)
  })
  self.run_pass(p, [w, g, u, y], (out + 7) / 8, 1)
}

///|
/// GPU argmax over n logits; cb receives the winning index. Two-phase
/// shared-memory reduction; readback is 4 bytes instead of the full
/// logits vector (~6ms/map saved per decode token).
pub fn Gpu::argmax(
  self : Gpu,
  x : GPUBuffer,
  n : Int,
  part_v : GPUBuffer,
  part_i : GPUBuffer,
  outb : GPUBuffer,
  cb : (Int) -> Unit,
) -> Unit {
  let chunk = 256
  let wgs = (n + chunk - 1) / chunk
  let p1 = self.pipeline("am1:\{n}", fn() { argmax1_shader(n, chunk) })
  self.run_pass(p1, [x, part_v, part_i], wgs, 1)
  let p2 = self.pipeline("am2:\{wgs}", fn() { argmax2_shader(wgs) })
  self.run_pass(p2, [part_v, part_i, outb], 1, 1)
  // 4-byte readback
  let readb = self.device.create_buffer(
    4,
    USAGE_MAP_READ | USAGE_COPY_DST,
    mapped_at_creation=false,
  )
  let enc = self.device.create_command_encoder()
  enc.copy_buffer_to_buffer(outb, 0, readb, 0, 4)
  self.queue.submit([enc.finish()])
  readb.map_async(MAP_MODE_READ, fn(_) {
    let r = readb.get_float_array()
    readb.unmap()
    cb(r[0].to_int())
  })
}

///|
/// Library 4D flash attention: q [B,H,Sq,D], k [B,HKV,Skv,D],
/// v [B,HKV,Skv,DV] -> out [B,H,Sq,DV]. One shader compile per
/// (shape, causal, scale) combo, cached. Requires D, DV <= 128; causal
/// needs Skv >= Sq (bottom-right mask). Grid: (ceil(Sq/64), B*H).
pub fn Gpu::attn_4d(
  self : Gpu,
  qb : GPUBuffer,
  kb : GPUBuffer,
  vb : GPUBuffer,
  outb : GPUBuffer,
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
  causal : Bool,
  scale : Double,
) -> Unit {
  if d > 128 || dv > 128 {
    abort("attn_4d: head dims must be <= 128")
  }
  if h % hkv != 0 {
    abort("attn_4d: kv_heads must divide heads")
  }
  if causal && skv < sq {
    abort("attn_4d: causal mask requires Skv >= Sq")
  }
  if b * h > 65535 {
    abort("attn_4d: B*H exceeds the dispatch grid.y limit")
  }
  let smem_need = 8 * (d + dv) * 4
  if smem_need > self.device.max_workgroup_storage() {
    abort(
      "attn_4d: needs \{smem_need}B workgroup storage, device allows \{self.device.max_workgroup_storage()}B",
    )
  }
  let p = self.pipeline(
    "attn4:\{b},\{h},\{hkv},\{sq},\{skv},\{d},\{dv},\{causal},\{scale}",
    fn() { attn_4d_shader(b, h, hkv, sq, skv, d, dv, causal, scale) },
  )
  let bg = self.device.make_bind_group(p, [qb, kb, vb, outb])
  let enc = self.device.create_command_encoder()
  let pass = enc.begin_compute_pass()
  pass.set_pipeline(p)
  pass.set_bind_group(0, bg)
  pass.dispatch_workgroups((sq + 63) / 64, b * h, 1)
  pass.end()
  self.queue.submit([enc.finish()])
}

///|
/// Naive 4D attention baseline: materializes the Sq x Skv score matrix in
/// global memory (two passes) instead of tiling. Same contract as
/// `attn_4d`; exists as the benchmark reference point.
pub fn Gpu::attn_naive_4d(
  self : Gpu,
  qb : GPUBuffer,
  kb : GPUBuffer,
  vb : GPUBuffer,
  outb : GPUBuffer,
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
  causal : Bool,
  scale : Double,
) -> Unit {
  if d > 128 || dv > 128 {
    abort("attn_naive_4d: head dims must be <= 128")
  }
  if h % hkv != 0 {
    abort("attn_naive_4d: kv_heads must divide heads")
  }
  if causal && skv < sq {
    abort("attn_naive_4d: causal mask requires Skv >= Sq")
  }
  if b * h > 65535 {
    abort("attn_naive_4d: B*H exceeds the dispatch grid.y limit")
  }
  let sb = self.alloc(b * h * sq * skv)
  let p1 = self.pipeline(
    "naive1:\{b},\{h},\{hkv},\{sq},\{skv},\{d},\{causal},\{scale}",
    fn() { attn_naive_scores_shader(b, h, hkv, sq, skv, d, causal, scale) },
  )
  let p2 = self.pipeline("naive2:\{h},\{hkv},\{sq},\{skv},\{dv},\{causal}", fn() {
    attn_naive_pv_shader(h, hkv, sq, skv, dv, causal)
  })
  let bg1 = self.device.make_bind_group(p1, [qb, kb, sb])
  let bg2 = self.device.make_bind_group(p2, [sb, vb, outb])
  let enc = self.device.create_command_encoder()
  let pass = enc.begin_compute_pass()
  pass.set_pipeline(p1)
  pass.set_bind_group(0, bg1)
  pass.dispatch_workgroups((sq + 63) / 64, b * h, 1)
  pass.set_pipeline(p2)
  pass.set_bind_group(0, bg2)
  pass.dispatch_workgroups((sq + 63) / 64, b * h, 1)
  pass.end()
  self.queue.submit([enc.finish()])
}
