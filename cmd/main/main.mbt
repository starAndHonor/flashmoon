///|
/// Demo: compare naive attention against flash attention on random inputs.
fn main {
  let n = 64 // query length
  let m = 64 // key/value length
  let d = 32 // head dim
  let dv = 32
  let rand = fn(i : Int, j : Int, k : Int) -> Float {
    (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
  }
  let mk = fn(rows : Int, cols : Int, k : Int) -> @nparray.NpArray {
    let data = FixedArray::make(rows * cols, 0.0F)
    for i in 0..<rows {
      for j in 0..<cols {
        data[i * cols + j] = rand(i, j, k)
      }
    }
    @tensor.Tensor::from_host(data, [rows, cols])
  }
  let q = mk(n, d, 0)
  let k = mk(m, d, 1)
  let v = mk(m, dv, 2)
  let naive = @flashmoon.naive_attention(q, k, v)
  let flash = @flashmoon.flash_attention(q, k, v, block_rows=16, block_cols=16)
  let da = naive.value().data
  let db = flash.value().data
  let mut max_diff = 0.0F
  for i in 0..<da.length() {
    let diff = (da[i] - db[i]).abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  println("attention output: [\{n}, \{dv}]")
  println("naive out[0][0] = \{da[0]}")
  println("flash out[0][0] = \{db[0]}")
  println("max |naive - flash| = \{max_diff}")
}
