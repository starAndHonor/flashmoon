///|
/// Whitebox: validate shape extraction and causal key-limit logic.

///|
test "extract_qkv returns 4D shape fields" {
  let q : @nparray.NpArray = @tensor.Tensor::from_host(
    FixedArray::make(2 * 4 * 3 * 8, 0.0F),
    [2, 4, 3, 8],
  )
  let k : @nparray.NpArray = @tensor.Tensor::from_host(
    FixedArray::make(2 * 2 * 5 * 8, 0.0F),
    [2, 2, 5, 8],
  )
  let v : @nparray.NpArray = @tensor.Tensor::from_host(
    FixedArray::make(2 * 2 * 5 * 6, 0.0F),
    [2, 2, 5, 6],
  )
  let cfg = AttnConfig::new(heads=4, kv_heads=2)
  let s = extract_qkv(q, k, v, cfg)
  assert_eq(s.b, 2)
  assert_eq(s.h, 4)
  assert_eq(s.sq, 3)
  assert_eq(s.skv, 5)
  assert_eq(s.d, 8)
  assert_eq(s.dv, 6)
}

///|
test "key_limit: bottom-right causal and non-causal" {
  // self-attention Sq == Skv (delta = 0): row i sees i+1 keys
  assert_eq(key_limit(true, 0, 4, 0), 1)
  assert_eq(key_limit(true, 3, 4, 0), 4)
  // cross/cached Sq < Skv (delta = 2): row 0 sees delta+1 keys
  assert_eq(key_limit(true, 0, 5, 2), 3)
  assert_eq(key_limit(true, 2, 5, 2), 5)
  // non-causal: all keys
  assert_eq(key_limit(false, 0, 5, 2), 5)
}
