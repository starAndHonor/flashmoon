///|
/// Deterministic pseudo-random value in [-1, 1) for reproducible tests.
fn pseudo_rand(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
fn make_tensor(
  rows : Int,
  cols : Int,
  f : (Int, Int) -> Float,
) -> @nparray_test.NpArray {
  let data = FixedArray::make(rows * cols, 0.0F)
  for i in 0..<rows {
    for j in 0..<cols {
      data[i * cols + j] = f(i, j)
    }
  }
  @tensor_test.Tensor::from_host(data, [rows, cols])
}

///|
fn max_abs_diff(a : @nparray_test.NpArray, b : @nparray_test.NpArray) -> Float {
  let da = a.value().data
  let db = b.value().data
  let mut mx = 0.0F
  for i in 0..<da.length() {
    let d = (da[i] - db[i]).abs()
    if d > mx {
      mx = d
    }
  }
  mx
}

///|
fn rand_qkv(
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
) -> (@nparray_test.NpArray, @nparray_test.NpArray, @nparray_test.NpArray) {
  (
    make_tensor(n, d, fn(i, j) { pseudo_rand(i, j, 0) }),
    make_tensor(m, d, fn(i, j) { pseudo_rand(i, j, 1) }),
    make_tensor(m, dv, fn(i, j) { pseudo_rand(i, j, 2) }),
  )
}

///|
test "flash matches naive: non-divisible blocks" {
  let (q, k, v) = rand_qkv(37, 53, 16, 24)
  let naive = @flash.naive_attention(q, k, v)
  let flash = @flash.flash_attention(q, k, v, block_rows=8, block_cols=13)
  let diff = max_abs_diff(naive, flash)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "flash matches naive: blocks larger than matrix" {
  let (q, k, v) = rand_qkv(3, 5, 4, 6)
  let naive = @flash.naive_attention(q, k, v)
  let flash = @flash.flash_attention(q, k, v, block_rows=32, block_cols=32)
  let diff = max_abs_diff(naive, flash)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "flash matches naive: 1x1 edge case" {
  let (q, k, v) = rand_qkv(1, 1, 2, 3)
  let naive = @flash.naive_attention(q, k, v)
  let flash = @flash.flash_attention(q, k, v, block_rows=1, block_cols=1)
  // single key: softmax degenerates to 1, output equals the single v row
  let expected = v.value().data
  let got = flash.value().data
  for c in 0..<3 {
    assert_true((got[c] - expected[c]).abs() < 1.0e-6)
  }
  assert_true(max_abs_diff(naive, flash) < 1.0e-6)
}

///|
test "zero scores give uniform average of v" {
  let n = 5
  let m = 7
  let dv = 4
  let q = make_tensor(n, 8, fn(_, _) { 0.0F })
  let k = make_tensor(m, 8, fn(_, _) { 0.0F })
  let v = make_tensor(m, dv, fn(i, j) { Float::from_int(i * dv + j) })
  let out = @flash.flash_attention(q, k, v, block_rows=2, block_cols=3)
  let data = out.value().data
  // every row must equal column means of v
  for i in 0..<n {
    for c in 0..<dv {
      let mut mean = 0.0F
      for j in 0..<m {
        mean += v.value().data[j * dv + c]
      }
      mean = mean / Float::from_int(m)
      assert_true((data[i * dv + c] - mean).abs() < 1.0e-6)
    }
  }
}
