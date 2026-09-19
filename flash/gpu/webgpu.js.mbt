///| WebGPU backend for flash attention (js target only).
///
/// Same 4D contract as `flash.flash_attention`: q [B,H,Sq,D],
/// k [B,HKV,Skv,D], v [B,HKV,Skv,DV] -> out [B,H,Sq,DV], MHA/GQA/MQA,
/// bottom-right causal mask, arbitrary head dims <= 128. Uploads the
/// tensors, dispatches one tiled online-softmax kernel per (batch, head,
/// 64-row block) and reads the result back asynchronously via `cb`.

///|
/// Flash attention on the GPU. Scale defaults to 1/sqrt(d) unless
/// `cfg.scale` is set. Requires D, DV <= 128.
pub fn flash_attention(
  g : @gpu.Gpu,
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  cfg : @flash.AttnConfig,
  cb : (@nparray.NpArray) -> Unit,
) -> Unit {
  let qd = q.value()
  let kd = k.value()
  let vd = v.value()
  if qd.dims.length() != 4 || kd.dims.length() != 4 || vd.dims.length() != 4 {
    abort("flash/gpu: expected 4D q/k/v [B, H, S, D]")
  }
  let b = qd.dims[0]
  let h = qd.dims[1]
  let sq = qd.dims[2]
  let d = qd.dims[3]
  let hkv = kd.dims[1]
  let skv = kd.dims[2]
  let dv = vd.dims[3]
  if kd.dims[0] != b ||
    vd.dims[0] != b ||
    vd.dims[1] != hkv ||
    vd.dims[2] != skv ||
    kd.dims[3] != d {
    abort("flash/gpu: shape mismatch")
  }
  if cfg.heads != h || cfg.kv_heads != hkv {
    abort("flash/gpu: AttnConfig heads/kv_heads must match tensor shapes")
  }
  let scale = match cfg.scale {
    Some(s) => s.to_double()
    None => 1.0 / d.to_double().sqrt()
  }
  let qb = g.upload(qd.data)
  let kb = g.upload(kd.data)
  let vb = g.upload(vd.data)
  let n_out = b * h * sq * dv
  let outb = g.alloc(n_out)
  g.attn_4d(qb, kb, vb, outb, b, h, hkv, sq, skv, d, dv, cfg.causal, scale)
  g.readback(outb, n_out, fn(data) {
    cb(@tensor.Tensor::from_host(data, [b, h, sq, dv]))
  })
}
