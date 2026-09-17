///|
/// Whitebox: validate internal shape extraction helpers.

///|
test "extract_qkv returns dims in q,k,v order" {
  let q : @nparray.NpArray = @tensor.Tensor::from_host(
    FixedArray::make(2 * 4, 0.0F),
    [2, 4],
  )
  let k : @nparray.NpArray = @tensor.Tensor::from_host(
    FixedArray::make(3 * 4, 0.0F),
    [3, 4],
  )
  let v : @nparray.NpArray = @tensor.Tensor::from_host(
    FixedArray::make(3 * 5, 0.0F),
    [3, 5],
  )
  let (_, _, _, n, m, d, dv) = extract_qkv(q, k, v)
  assert_eq(n, 2)
  assert_eq(m, 3)
  assert_eq(d, 4)
  assert_eq(dv, 5)
}
