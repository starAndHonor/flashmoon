///|
/// Minimal `flash` library example (CPU: wasm f32x4 SIMD / native scalar).
///
/// Shows the 4D contract — q [B,H,Sq,D], k/v [B,HKV,Skv,DV] -> [B,H,Sq,DV] —
/// with GQA (`hkv < h`) plus a bottom-right causal mask, and checks the tiled
/// kernel against the naive reference that ships in the same package.
///
/// Run: moon run examples/attn_cpu --target wasm --release

///|
/// Deterministic pseudo-random value in [-1, 1) (example stand-in for real data).
fn rand_val(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
/// Build a [b, h, rows, cols] tensor filled with the deterministic pattern.
fn make_tensor(
  b : Int,
  h : Int,
  rows : Int,
  cols : Int,
  k : Int,
) -> @nparray.NpArray {
  let data = FixedArray::make(b * h * rows * cols, 0.0F)
  for i in 0..<(b * h * rows) {
    for j in 0..<cols {
      data[i * cols + j] = rand_val(i, j, k)
    }
  }
  @tensor.Tensor::from_host(data, [b, h, rows, cols])
}

///|
fn main {
  // 2 batches, 4 query heads over 2 kv heads (GQA), 64x64 sequence, head dim 32.
  let b = 2
  let h = 4
  let hkv = 2
  let sq = 64
  let skv = 64
  let d = 32
  let dv = 32
  let q = make_tensor(b, h, sq, d, 0)
  let k = make_tensor(b, hkv, skv, d, 1)
  let v = make_tensor(b, hkv, skv, dv, 2)
  let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal=true)
  // Tiled online-softmax kernel (wasm SIMD / native scalar).
  let out = @flash.flash_attention(q, k, v, cfg, block_rows=64, block_cols=64)
  // Naive reference over the same inputs (materializes the score matrix).
  let ref = @flash.naive_attention(q, k, v, cfg)
  let a = out.value().data
  let r = ref.value().data
  let mut max_diff = 0.0F
  for i in 0..<a.length() {
    let diff = (a[i] - r[i]).abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  println("out dims   = \{out.value().dims}")
  println("out[0]     = \{a[0]}")
  println("naive[0]   = \{r[0]}")
  println("max |flash - naive| = \{max_diff}")
}
