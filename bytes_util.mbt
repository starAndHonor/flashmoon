///|
/// Little-endian f32 access over byte buffers (wasm linear memory layout).

///|
fn get_f32(b : FixedArray[Byte], idx : Int) -> Float {
  let o = idx * 4
  let x = b.unsafe_get(o).to_int() |
    (b.unsafe_get(o + 1).to_int() << 8) |
    (b.unsafe_get(o + 2).to_int() << 16) |
    (b.unsafe_get(o + 3).to_int() << 24)
  Float::reinterpret_from_int(x)
}

///|
fn set_f32(b : FixedArray[Byte], idx : Int, x : Float) -> Unit {
  let v = x.reinterpret_as_int()
  let o = idx * 4
  b.unsafe_set(o, (v & 0xff).to_byte())
  b.unsafe_set(o + 1, ((v >> 8) & 0xff).to_byte())
  b.unsafe_set(o + 2, ((v >> 16) & 0xff).to_byte())
  b.unsafe_set(o + 3, ((v >> 24) & 0xff).to_byte())
}

///|
fn floats_to_bytes(f : FixedArray[Float]) -> FixedArray[Byte] {
  let b = FixedArray::make(f.length() * 4, b'\x00')
  for i in 0..<f.length() {
    set_f32(b, i, f.unsafe_get(i))
  }
  b
}
