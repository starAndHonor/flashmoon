///|
/// SafeTensors reader: 8-byte LE header length, JSON header, raw data.
/// Only the dtypes Qwen3-0.6B needs (BF16, F32) are decoded.

///|
pub struct TensorInfo {
  dtype : String
  shape : Array[Int]
  begin : Int
  end : Int
}

///|
pub struct SafeTensors {
  data : Bytes
  entries : Map[String, TensorInfo]
}

///|
fn le_u64(b : Bytes, off : Int) -> Int64 {
  let mut v = 0L
  for i in 0..<8 {
    v = v | (b[off + i].to_int().to_int64() << (i * 8))
  }
  v
}

///|
fn le_u16(b : Bytes, off : Int) -> Int {
  b[off].to_int() | (b[off + 1].to_int() << 8)
}

///|
fn le_u32(b : Bytes, off : Int) -> Int {
  b[off].to_int() |
  (b[off + 1].to_int() << 8) |
  (b[off + 2].to_int() << 16) |
  (b[off + 3].to_int() << 24)
}

///|
pub fn SafeTensors::parse(data : Bytes) -> SafeTensors raise Error {
  let hlen = le_u64(data, 0).to_int()
  let header : String = @utf8.decode(data[8:8 + hlen]) catch {
    _ => raise Failure("safetensors header is not utf8")
  }
  let json = @json.parse(header)
  guard json is Object(obj) else { raise Failure("header must be an object") }
  let entries : Map[String, TensorInfo] = Map([])
  for name, meta in obj {
    if name == "__metadata__" {
      continue
    }
    guard meta
      is {
        "dtype": String(dtype),
        "shape": Array(shape),
        "data_offsets": Array([Number(begin, ..), Number(end, ..)]),
        ..
      } else {
      raise Failure("bad tensor entry: \{name}")
    }
    let dims : Array[Int] = []
    for d in shape {
      guard d is Number(v, ..) else { raise Failure("bad shape: \{name}") }
      dims.push(v.to_int())
    }
    entries[name] = {
      dtype,
      shape: dims,
      begin: 8 + hlen + begin.to_int(),
      end: 8 + hlen + end.to_int(),
    }
  }
  { data, entries }
}

///|
/// Decode a tensor to f32 (BF16 upcast: top 16 bits of f32).
pub fn SafeTensors::get_f32(
  self : SafeTensors,
  name : String,
) -> (FixedArray[Float], Array[Int]) raise Error {
  guard self.entries.get(name) is Some(info) else {
    raise Failure("tensor not found: \{name}")
  }
  let n = info.shape.fold(init=1, fn(acc, d) { acc * d })
  let out : FixedArray[Float] = FixedArray::make(n, (0.0 : Float))
  match info.dtype {
    "BF16" =>
      for i in 0..<n {
        let bits = le_u16(self.data, info.begin + i * 2) << 16
        out[i] = Float::reinterpret_from_int(bits)
      }
    "F32" =>
      for i in 0..<n {
        out[i] = Float::reinterpret_from_int(
          le_u32(self.data, info.begin + i * 4),
        )
      }
    dt => raise Failure("unsupported dtype \{dt} for \{name}")
  }
  (out, info.shape)
}
