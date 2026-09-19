///|
/// Minimal WebGPU example: the same 4D attention contract as
/// `examples/attn_cpu`, dispatched on the GPU.
///
/// Two levels are shown:
///   1. `@flashgpu.flash_attention` — library wrapper (upload, dispatch,
///      readback, result as an NpArray);
///   2. `@gpu.Gpu::attn_4d` — device level, for timing or when the tensors
///      already live on the GPU.
///
/// Run: moon build --target js && deno run --allow-read scripts/attn_gpu_host.js

///|
extern "js" fn post(s : String) -> Unit =
  #|(s) => { fetch('/result', { method: 'POST', body: s }); }

///|
fn rand_val(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
fn make_data(
  b : Int,
  h : Int,
  rows : Int,
  cols : Int,
  k : Int,
) -> FixedArray[Float] {
  let data = FixedArray::make(b * h * rows * cols, 0.0F)
  for i in 0..<(b * h * rows) {
    for j in 0..<cols {
      data[i * cols + j] = rand_val(i, j, k)
    }
  }
  data
}

///|
fn as_tensor4(
  data : FixedArray[Float],
  b : Int,
  h : Int,
  rows : Int,
  cols : Int,
) -> @nparray.NpArray {
  @tensor.Tensor::from_host(data, [b, h, rows, cols])
}

///|
fn max_diff(a : FixedArray[Float], b : FixedArray[Float]) -> Float {
  let mut mx = 0.0F
  for i in 0..<a.length() {
    let d = (a.unsafe_get(i) - b.unsafe_get(i)).abs()
    if d > mx {
      mx = d
    }
  }
  mx
}

///|
fn main {
  let b = 1
  let h = 8
  let hkv = 2
  let sq = 128
  let skv = 128
  let d = 64
  let dv = 64
  let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal=true)
  let q = as_tensor4(make_data(b, h, sq, d, 0), b, h, sq, d)
  let k = as_tensor4(make_data(b, hkv, skv, d, 1), b, hkv, skv, d)
  let v = as_tensor4(make_data(b, hkv, skv, dv, 2), b, hkv, skv, dv)
  // CPU oracle for the same inputs (identical math, independent kernel).
  let want = @flash.naive_attention(q, k, v, cfg)
  @gpu.Gpu::init(
    fn(g) {
      // (1) Library wrapper: handles upload + dispatch + readback.
      @flashgpu.flash_attention(g, q, k, v, cfg, fn(out) {
        let dims = out.value().dims
        post(
          "flash/gpu  out dims=[\{dims[0]}, \{dims[1]}, \{dims[2]}, \{dims[3]}] max|gpu - cpu|=\{max_diff(out.value().data, want.value().data)}",
        )
        // (2) Device level: tensors already on the GPU, no per-call upload.
        let qb = g.upload(q.value().data)
        let kb = g.upload(k.value().data)
        let vb = g.upload(v.value().data)
        let outb = g.alloc(b * h * sq * dv)
        let scale = 1.0 / d.to_double().sqrt()
        g.attn_4d(qb, kb, vb, outb, b, h, hkv, sq, skv, d, dv, true, scale)
        g.readback(outb, b * h * sq * dv, fn(got) {
          post(
            "gpu attn_4d  max|gpu - cpu|=\{max_diff(got, want.value().data)}",
          )
        })
      })
    },
    log=post,
  )
}
