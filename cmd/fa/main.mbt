///|
/// Demo: batched GQA causal attention — naive reference vs flash kernel
/// on random 4D inputs.
fn main {
  let b = 2 // batch
  let h = 4 // query heads
  let hkv = 2 // key/value heads (GQA)
  let sq = 64 // query length
  let skv = 64 // key/value length
  let d = 32 // head dim
  let rand = fn(i : Int, j : Int, k : Int) -> Float {
    (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
  }
  let mk = fn(bh : Int, rows : Int, cols : Int, k : Int) -> @nparray.NpArray {
    let data = FixedArray::make(bh * rows * cols, 0.0F)
    for i in 0..<(bh * rows) {
      for j in 0..<cols {
        data[i * cols + j] = rand(i, j, k)
      }
    }
    @tensor.Tensor::from_host(data, [b, bh / b, rows, cols])
  }
  let q = mk(b * h, sq, d, 0)
  let k = mk(b * hkv, skv, d, 1)
  let v = mk(b * hkv, skv, d, 2)
  let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal=true)
  let naive = @flash.naive_attention(q, k, v, cfg)
  let flash = @flash.flash_attention(q, k, v, cfg, block_rows=16, block_cols=16)
  let da = naive.value().data
  let db = flash.value().data
  let mut max_diff = 0.0F
  for i in 0..<da.length() {
    let diff = (da[i] - db[i]).abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  println(
    "attention: b=\{b} h=\{h} hkv=\{hkv} sq=\{sq} skv=\{skv} d=\{d} causal",
  )
  println("naive out[0] = \{da[0]}")
  println("flash out[0] = \{db[0]}")
  println("max |naive - flash| = \{max_diff}")
}
