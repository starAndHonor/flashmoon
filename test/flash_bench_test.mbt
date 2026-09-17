///|
/// Benchmarks: naive vs flash at various tile sizes.

///|
fn bench_qkv(
  n : Int,
  m : Int,
  d : Int,
) -> (@nparray_test.NpArray, @nparray_test.NpArray, @nparray_test.NpArray) {
  let rand = fn(i : Int, j : Int, k : Int) -> Float {
    (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
  }
  let mk = fn(rows : Int, cols : Int, k : Int) -> @nparray_test.NpArray {
    let data = FixedArray::make(rows * cols, 0.0F)
    for i in 0..<rows {
      for j in 0..<cols {
        data[i * cols + j] = rand(i, j, k)
      }
    }
    @tensor_test.Tensor::from_host(data, [rows, cols])
  }
  (mk(n, d, 0), mk(m, d, 1), mk(m, d, 2))
}

///|
test "bench attention 512x512 d=64" (b : @bench.T) {
  let (q, k, v) = bench_qkv(512, 512, 64)
  let naive = fn() {
    let out = @flashmoon.naive_attention(q, k, v)
    b.keep(out)
  }
  b.bench(name="naive", naive)
  let flash16 = fn() {
    let out = @flashmoon.flash_attention(q, k, v, block_rows=16, block_cols=16)
    b.keep(out)
  }
  b.bench(name="flash b16", flash16)
  let flash32 = fn() {
    let out = @flashmoon.flash_attention(q, k, v, block_rows=32, block_cols=32)
    b.keep(out)
  }
  b.bench(name="flash b32", flash32)
  let flash64 = fn() {
    let out = @flashmoon.flash_attention(q, k, v, block_rows=64, block_cols=64)
    b.keep(out)
  }
  b.bench(name="flash b64", flash64)
  let flash128 = fn() {
    let out = @flashmoon.flash_attention(
      q,
      k,
      v,
      block_rows=128,
      block_cols=128,
    )
    b.keep(out)
  }
  b.bench(name="flash b128", flash128)
}
