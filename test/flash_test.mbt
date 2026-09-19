///|
/// Deterministic pseudo-random value in [-1, 1) for reproducible tests.
fn pseudo_rand(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
/// Fill a 4D [b, h, s, c] tensor with a deterministic pattern.
fn make_tensor4(
  b : Int,
  h : Int,
  s : Int,
  c : Int,
  f : (Int, Int) -> Float,
) -> @nparray_test.NpArray {
  let data = FixedArray::make(b * h * s * c, 0.0F)
  for i in 0..<(b * h * s) {
    for j in 0..<c {
      data[i * c + j] = f(i, j)
    }
  }
  @tensor_test.Tensor::from_host(data, [b, h, s, c])
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
/// Random q [b,h,sq,d], k [b,hkv,skv,d], v [b,hkv,skv,dv].
fn rand_qkv(
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
) -> (@nparray_test.NpArray, @nparray_test.NpArray, @nparray_test.NpArray) {
  (
    make_tensor4(b, h, sq, d, fn(i, j) { pseudo_rand(i, j, 0) }),
    make_tensor4(b, hkv, skv, d, fn(i, j) { pseudo_rand(i, j, 1) }),
    make_tensor4(b, hkv, skv, dv, fn(i, j) { pseudo_rand(i, j, 2) }),
  )
}

///|
/// flash vs naive over a grid of shapes and configs; returns max diff.
fn check_case(
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
  causal : Bool,
  br : Int,
  bc : Int,
) -> Float {
  let (q, k, v) = rand_qkv(b, h, hkv, sq, skv, d, dv)
  let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal~)
  let naive = @flash.naive_attention(q, k, v, cfg)
  let flash = @flash.flash_attention(q, k, v, cfg, block_rows=br, block_cols=bc)
  max_abs_diff(naive, flash)
}

///|
test "flash matches naive: non-divisible blocks" {
  let diff = check_case(1, 1, 1, 37, 53, 16, 24, false, 8, 13)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "flash matches naive: blocks larger than matrix" {
  let diff = check_case(1, 1, 1, 3, 5, 4, 6, false, 32, 32)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "flash matches naive: 1x1 edge case" {
  let (q, k, v) = rand_qkv(1, 1, 1, 1, 1, 2, 3)
  let cfg = @flash.AttnConfig::new(heads=1)
  let naive = @flash.naive_attention(q, k, v, cfg)
  let flash = @flash.flash_attention(q, k, v, cfg, block_rows=1, block_cols=1)
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
  let q = make_tensor4(1, 1, n, 8, fn(_, _) { 0.0F })
  let k = make_tensor4(1, 1, m, 8, fn(_, _) { 0.0F })
  let v = make_tensor4(1, 1, m, dv, fn(i, j) { Float::from_int(i * dv + j) })
  let cfg = @flash.AttnConfig::new(heads=1)
  let out = @flash.flash_attention(q, k, v, cfg, block_rows=2, block_cols=3)
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

///|
test "property: batched GQA with causal mask" {
  // b=2, 4 q-heads over 2 kv-heads, causal self-attention
  let diff = check_case(2, 4, 2, 37, 37, 16, 24, true, 8, 13)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "property: MQA cross-attention with cache (Sq < Skv)" {
  // 3 q-heads over 1 kv-head, causal with 51 cached positions
  let diff = check_case(1, 3, 1, 5, 56, 16, 8, true, 4, 9)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "property: non-causal cross attention, odd dims" {
  // head dim 13 and value dim 7: no SIMD-friendly alignment anywhere
  let diff = check_case(2, 2, 2, 11, 17, 13, 7, false, 5, 3)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "property: causal self-attention, odd dims, tiny blocks" {
  let diff = check_case(1, 2, 1, 23, 23, 13, 7, true, 1, 1)
  assert_true(diff < 1.0e-4, msg="max diff \{diff}")
}

///|
test "causal mask zeroes future attention exactly" {
  // With causal masking, row 0 attends only key 0: out row 0 == v row 0.
  let (q, k, v) = rand_qkv(1, 1, 1, 4, 4, 8, 8)
  let cfg = @flash.AttnConfig::new(heads=1, causal=true)
  let out = @flash.flash_attention(q, k, v, cfg)
  let data = out.value().data
  let vd = v.value().data
  for c in 0..<8 {
    assert_true((data[c] - vd[c]).abs() < 1.0e-6)
  }
}
