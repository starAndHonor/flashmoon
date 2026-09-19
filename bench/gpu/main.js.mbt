///|
/// GPU attention benchmark (JS/WebGPU target): a naive kernel that
/// materializes the Sq x Skv score matrix in global memory vs the tiled
/// flash kernel that never materializes it. Both share the 4D contract, so
/// the comparison is algorithm-only. Results are posted via
/// fetch("/result") (scripts/bench_gpu_host.js runs this under Deno).
///
/// Run: moon build --target js && deno run --allow-read scripts/bench_gpu_host.js

///|
extern "js" fn post(s : String) -> Unit =
  #|(s) => { fetch('/result', { method: 'POST', body: s }); }

///|
extern "js" fn now_ms() -> Double =
  #|() => performance.now()

///|
/// Timing rounds per kernel (min is reported).
const REPS : Int = 3

///|
/// Dispatches per round for a scenario, so that the fixed sync round-trip
/// (~ms under Deno/wgpu) does not swamp the kernel time of small shapes.
fn iters_for(b : Int, h : Int, sq : Int, skv : Int) -> Int {
  let work = b * h * sq * skv
  if work >= 8_000_000 {
    1
  } else {
    let n = 8_000_000 / (if work == 0 { 1 } else { work })
    if n > 50 {
      50
    } else {
      n
    }
  }
}

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
      data.unsafe_set(i * cols + j, rand_val(i, j, k))
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
fn fmt2(x : Double) -> String {
  let neg = x < 0.0
  let v = (x.abs() * 100.0 + 0.5).floor().to_int64()
  let int_part = v / 100
  let frac = (v % 100).to_int()
  let frac_s = if frac < 10 { "0\{frac}" } else { "\{frac}" }
  if neg {
    "-\{int_part}.\{frac_s}"
  } else {
    "\{int_part}.\{frac_s}"
  }
}

///|
fn fmt_err(x : Float) -> String {
  if x == 0.0F {
    return "0"
  }
  let mut v = x.abs().to_double()
  let mut e = 0
  while v >= 10.0 {
    v /= 10.0
    e += 1
  }
  while v < 1.0 {
    v *= 10.0
    e -= 1
  }
  "\{fmt2(v)}e\{e}"
}

///|
fn pad(s : String, w : Int) -> String {
  let buf = StringBuilder::new()
  buf.write_string(s)
  for _ in 0..<(w - s.length()) {
    buf.write_char(' ')
  }
  buf.to_string()
}

///|
/// Dense-equivalent attention FLOPs (causal runs do ~half the work).
fn flops(b : Int, h : Int, sq : Int, skv : Int, d : Int, dv : Int) -> Double {
  2.0 *
  b.to_double() *
  h.to_double() *
  sq.to_double() *
  skv.to_double() *
  (d.to_double() + dv.to_double())
}

///|
/// Min per-dispatch GPU time over `reps` rounds of `iters` dispatches, in
/// milliseconds.
fn time_gpu(
  g : @gpu.Gpu,
  reps : Int,
  iters : Int,
  k : Int,
  best : Double,
  f : () -> Unit,
  cb : (Double) -> Unit,
) -> Unit {
  if k >= reps {
    cb(best)
    return
  }
  let t0 = now_ms()
  for _ in 0..<iters {
    f()
  }
  g.sync(fn(_) {
    let dt = (now_ms() - t0) / iters.to_double()
    time_gpu(g, reps, iters, k + 1, if dt < best { dt } else { best }, f, cb)
  })
}

///|
fn gflops(
  ms : Double,
  b : Int,
  h : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
) -> String {
  fmt2(flops(b, h, sq, skv, d, dv) / ms / 1.0e6)
}

///|
/// One scenario: time naive and flash on the GPU, then check flash's output
/// against the CPU naive oracle. `next` chains to the following scenario.
fn run_case(
  g : @gpu.Gpu,
  label : String,
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
  causal : Bool,
  next : () -> Unit,
) -> Unit {
  let qd = make_data(b, h, sq, d, 0)
  let kd = make_data(b, hkv, skv, d, 1)
  let vd = make_data(b, hkv, skv, dv, 2)
  let qb = g.upload(qd)
  let kb = g.upload(kd)
  let vb = g.upload(vd)
  let n_out = b * h * sq * dv
  let outn = g.alloc(n_out)
  let outf = g.alloc(n_out)
  let scale = 1.0 / d.to_double().sqrt()
  let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal~)
  let oracle = @flash.naive_attention(
      as_tensor4(qd, b, h, sq, d),
      as_tensor4(kd, b, hkv, skv, d),
      as_tensor4(vd, b, hkv, skv, dv),
      cfg,
    ).value().data
  // warmup both kernels (shader compile + cache)
  g.attn_naive_4d(qb, kb, vb, outn, b, h, hkv, sq, skv, d, dv, causal, scale)
  g.attn_4d(qb, kb, vb, outf, b, h, hkv, sq, skv, d, dv, causal, scale)
  let iters = iters_for(b, h, sq, skv)
  g.sync(fn(_) {
    time_gpu(
      g,
      REPS,
      iters,
      0,
      1.0e300,
      fn() {
        g.attn_naive_4d(
          qb, kb, vb, outn, b, h, hkv, sq, skv, d, dv, causal, scale,
        )
      },
      fn(tn) {
        time_gpu(
          g,
          REPS,
          iters,
          0,
          1.0e300,
          fn() {
            g.attn_4d(
              qb, kb, vb, outf, b, h, hkv, sq, skv, d, dv, causal, scale,
            )
          },
          fn(tf) {
            g.readback(outf, n_out, fn(got) {
              let diff = max_diff(got, oracle)
              let shape = "\{b}x\{h}/\{hkv} \{sq}x\{skv}x\{d}"
              post(
                pad(label, 22) +
                pad(shape, 18) +
                pad(fmt2(tn), 11) +
                pad(fmt2(tf), 11) +
                pad(fmt2(tn / tf) + "x", 9) +
                pad(gflops(tf, b, h, sq, skv, d, dv), 11) +
                fmt_err(diff),
              )
              next()
            })
          },
        )
      },
    )
  })
}

///|
fn run_cases(
  g : @gpu.Gpu,
  cases : Array[(String, Int, Int, Int, Int, Int, Int, Int, Bool)],
  idx : Int,
) -> Unit {
  if idx >= cases.length() {
    return
  }
  let c = cases[idx]
  run_case(g, c.0, c.1, c.2, c.3, c.4, c.5, c.6, c.7, c.8, fn() {
    run_cases(g, cases, idx + 1)
  })
}

///|
fn main {
  let cases : Array[(String, Int, Int, Int, Int, Int, Int, Int, Bool)] = [
    ("chat prompt", 1, 8, 8, 256, 256, 64, 64, true),
    ("long prefill 2k", 1, 8, 8, 2048, 2048, 64, 64, true),
    ("decode step (KV 2k)", 1, 8, 8, 1, 2048, 64, 64, true),
    ("GQA decode (KV 4k)", 1, 8, 2, 1, 4096, 64, 64, true),
    ("MQA decode (KV 4k)", 1, 8, 1, 1, 4096, 64, 64, true),
    ("batched prefill b4", 4, 8, 8, 256, 256, 64, 64, true),
    ("cached cross-attn", 1, 8, 8, 64, 1024, 64, 64, true),
    ("wide head d=128", 1, 8, 8, 1024, 1024, 128, 128, true),
    ("odd dims d80/dv96", 1, 4, 2, 512, 512, 80, 96, true),
    ("seqlen 512 h1", 1, 1, 1, 512, 512, 64, 64, false),
    ("seqlen 1024 h1", 1, 1, 1, 1024, 1024, 64, 64, false),
    ("seqlen 2048 h1", 1, 1, 1, 2048, 2048, 64, 64, false),
    ("seqlen 4096 h1", 1, 1, 1, 4096, 4096, 64, 64, false),
  ]
  post("# GPU attention: naive (materialized scores) vs tiled flash")
  post(
    pad("scenario", 22) +
    pad("shape (BxH/Hkv SqxSkvxD)", 18) +
    pad("naive ms", 11) +
    pad("flash ms", 11) +
    pad("speedup", 9) +
    pad("flash GF/s", 11) +
    "max diff vs CPU",
  )
  @gpu.Gpu::init(fn(g) { g.sync(fn(_) { run_cases(g, cases, 0) }) }, log=post)
}
