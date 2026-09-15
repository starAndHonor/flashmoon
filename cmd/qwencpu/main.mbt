///|
/// Qwen3-0.6B CPU forward: compare against HF reference logits.
///
/// Usage: moon run cmd/qwen --target native -- <model.safetensors> <ref_input.json> <ref_logits.f32>

///|
fn le_f32(b : Bytes, off : Int) -> Float {
  Float::reinterpret_from_int(
    b[off].to_int() |
    (b[off + 1].to_int() << 8) |
    (b[off + 2].to_int() << 16) |
    (b[off + 3].to_int() << 24),
  )
}

///|
fn run() -> Unit raise Error {
  let args = @env.args()
  guard args.length() >= 4 else {
    println("usage: qwen <model.safetensors> <ref_input.json> <ref_logits.f32>")
    panic()
  }
  let st = @qwen3.SafeTensors::parse(@fs.read_file_to_bytes(args[1]))
  let model = @qwen3.Qwen3::load(st)
  println("weights loaded")
  // reference input tokens
  let input_json = @json.parse(@fs.read_file_to_string(args[2]))
  guard input_json is Array(ids_json) else { panic() }
  let ids : Array[Int] = []
  for id in ids_json {
    guard id is Number(v, ..) else { panic() }
    ids.push(v.to_int())
  }
  // forward
  let logits = model.forward(ids)
  println("forward done")
  // compare with reference
  let refb = @fs.read_file_to_bytes(args[3])
  let mut max_diff : Double = 0.0
  let mut argmax = 0
  let mut best : Float = -3.4e38
  for i in 0..<model.vocab {
    let d = (logits[i] - le_f32(refb, i * 4)).to_double().abs()
    if d > max_diff {
      max_diff = d
    }
    if logits[i] > best {
      best = logits[i]
      argmax = i
    }
  }
  println("max_diff=\{max_diff}")
  println("argmax=\{argmax} (expect 12095 ' Paris')")
  if max_diff < 0.1 && argmax == 12095 {
    println("MATCH")
  } else {
    println("MISMATCH")
  }
}

///|
fn main {
  run() catch {
    e => println("error: \{e}")
  }
}
