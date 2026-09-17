///|
/// Scalar fallback kernels for non-wasm targets (same signatures as
/// kernel_wasm.mbt). Operates on little-endian f32 byte buffers.

///|
fn kdot(
  q : FixedArray[Byte],
  qi : Int,
  k : FixedArray[Byte],
  ki : Int,
  n : Int,
) -> Float {
  let mut s = 0.0F
  for t in 0..<n {
    s += get_f32(q, qi + t) * get_f32(k, ki + t)
  }
  s
}

///|
fn kaxpy(
  acc : FixedArray[Byte],
  ab : Int,
  p : Float,
  v : FixedArray[Byte],
  vb : Int,
  n : Int,
) -> Unit {
  for c in 0..<n {
    set_f32(acc, ab + c, get_f32(acc, ab + c) + p * get_f32(v, vb + c))
  }
}

///|
fn kscale(acc : FixedArray[Byte], ab : Int, alpha : Float, n : Int) -> Unit {
  for c in 0..<n {
    set_f32(acc, ab + c, get_f32(acc, ab + c) * alpha)
  }
}
