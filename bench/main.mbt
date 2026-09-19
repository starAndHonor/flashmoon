///|
/// FlashAttention evaluation harness: naive reference vs flash library.
///
/// Section 1 is scenario-driven — common LLM attention shapes (chat prompt,
/// long prefill, decode step with KV cache, GQA/MQA decode, batched prefill,
/// cached cross-attention, wide head dim, odd head dims). Sections 2-4 sweep
/// sequence length, head dim and tile size.
///
/// Reports wall-clock (min of reps, release), dense-equivalent throughput in
/// GFLOP/s, and numerical agreement with the naive reference.
///
/// Run: moon run bench --target wasm --release
///      moon run bench --target native --release

///|
/// Deterministic pseudo-random value in [-1, 1).
fn rand_val(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
/// q [b,hq,n,d], k [b,hkv,m,d], v [b,hkv,m,dv].
fn make_qkv(
  b : Int,
  hq : Int,
  hkv : Int,
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
) -> (@nparray.NpArray, @nparray.NpArray, @nparray.NpArray) {
  let mk = fn(h : Int, rows : Int, cols : Int, k : Int) -> @nparray.NpArray {
    let data = FixedArray::make(b * h * rows * cols, 0.0F)
    for i in 0..<(b * h * rows) {
      for j in 0..<cols {
        data.unsafe_set(i * cols + j, rand_val(i, j, k))
      }
    }
    @tensor.Tensor::from_host(data, [b, h, rows, cols])
  }
  (mk(hq, n, d, 0), mk(hkv, m, d, 1), mk(hkv, m, dv, 2))
}

///|
fn max_abs_diff(a : @nparray.NpArray, b : @nparray.NpArray) -> Float {
  let da = a.value().data
  let db = b.value().data
  let mut mx = 0.0F
  for i in 0..<da.length() {
    let d = (da.unsafe_get(i) - db.unsafe_get(i)).abs()
    if d > mx {
      mx = d
    }
  }
  mx
}

///|
/// Min wall-clock over `reps` runs after 1 warmup, in microseconds.
fn timed_us(f : () -> Unit, reps : Int) -> Double {
  f() // warmup
  let mut best = 1.0e300
  for _ in 0..<reps {
    let t0 = @cbench.monotonic_clock_start()
    f()
    let us = @cbench.monotonic_clock_end(t0)
    if us < best {
      best = us
    }
  }
  best
}

///|
/// Format a double with 2 decimals (fixed-point, zero-padded).
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
/// Format a float in scientific-ish notation for tiny error values.
fn fmt_err(x : Float) -> String {
  if x == 0.0F {
    return "0"
  }
  // normalize to [1, 10) with decimal exponent
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
/// Dense-equivalent attention FLOPs: QK^T (2*Sq*Skv*D) + PV (2*Sq*Skv*Dv)
/// per (batch, head). Causal runs do ~half this work.
fn flops(b : Int, h : Int, n : Int, m : Int, d : Int, dv : Int) -> Double {
  2.0 *
  b.to_double() *
  h.to_double() *
  n.to_double() *
  m.to_double() *
  (d.to_double() + dv.to_double())
}

///|
fn gflops(
  us : Double,
  b : Int,
  h : Int,
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
) -> String {
  fmt2(flops(b, h, n, m, d, dv) / us / 1000.0)
}

///|
/// One scenario: label + (b, hq, hkv, sq, skv, d, dv, causal).
fn scenario(
  reps : Int,
  label : String,
  b : Int,
  hq : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
  causal : Bool,
) -> Unit {
  let (q, k, v) = make_qkv(b, hq, hkv, sq, skv, d, dv)
  let cfg = @flash.AttnConfig::new(heads=hq, kv_heads=hkv, causal~)
  let t_naive = timed_us(
    fn() { @flash.naive_attention(q, k, v, cfg) |> ignore },
    reps,
  )
  let t_flash = timed_us(
    fn() { @flash.flash_attention(q, k, v, cfg) |> ignore },
    reps,
  )
  let diff = max_abs_diff(
    @flash.naive_attention(q, k, v, cfg),
    @flash.flash_attention(q, k, v, cfg),
  )
  println(
    pad(label, 26) +
    pad("\{b}x\{hq}/\{hkv} \{sq}x\{skv}x\{d}", 18) +
    pad(fmt2(t_naive), 11) +
    pad(fmt2(t_flash), 11) +
    pad(fmt2(t_naive / t_flash) + "x", 9) +
    pad(gflops(t_flash, b, hq, sq, skv, d, dv), 11) +
    fmt_err(diff),
  )
}

///|
/// Common attention scenarios for LLM inference, naive vs flash.
fn sweep_scenarios(reps : Int) -> Unit {
  println("## 1. Common scenarios (naive vs flash, tiles 64x64, causal)")
  println("")
  println(
    pad("scenario", 26) +
    pad("shape (BxH/Hkv SqxSkvxD)", 18) +
    pad("naive us", 11) +
    pad("flash us", 11) +
    pad("speedup", 9) +
    pad("flash GF/s", 11) +
    "max diff",
  )
  scenario(reps, "chat prompt", 1, 8, 8, 256, 256, 64, 64, true)
  scenario(reps, "long prefill 2k", 1, 8, 8, 2048, 2048, 64, 64, true)
  scenario(reps, "decode step (KV 2k)", 1, 8, 8, 1, 2048, 64, 64, true)
  scenario(reps, "GQA decode (KV 4k)", 1, 8, 2, 1, 4096, 64, 64, true)
  scenario(reps, "MQA decode (KV 4k)", 1, 8, 1, 1, 4096, 64, 64, true)
  scenario(reps, "batched prefill b4", 4, 8, 8, 256, 256, 64, 64, true)
  scenario(reps, "cached cross-attn", 1, 8, 8, 64, 1024, 64, 64, true)
  scenario(reps, "wide head d=128", 1, 8, 8, 1024, 1024, 128, 128, true)
  scenario(reps, "odd dims d80/dv96", 1, 4, 2, 512, 512, 80, 96, true)
}

///|
/// Sequence-length sweep, naive vs flash (default 64x64 tiles), non-causal.
fn sweep_seqlen(reps : Int) -> Unit {
  println("")
  println("## 2. Sequence length sweep (n = m, d = dv = 64, non-causal)")
  println("")
  println(
    pad("size", 10) +
    pad("naive us", 12) +
    pad("flash us", 12) +
    pad("speedup", 10) +
    pad("naive GF/s", 13) +
    pad("flash GF/s", 13) +
    "max diff",
  )
  let cfg = @flash.AttnConfig::new(heads=1)
  for s in [128, 256, 512, 1024, 2048, 4096] {
    let (q, k, v) = make_qkv(1, 1, 1, s, s, 64, 64)
    let t_naive = timed_us(
      fn() { @flash.naive_attention(q, k, v, cfg) |> ignore },
      reps,
    )
    let t_flash = timed_us(
      fn() { @flash.flash_attention(q, k, v, cfg) |> ignore },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q, k, v, cfg),
      @flash.flash_attention(q, k, v, cfg),
    )
    println(
      pad("\{s}x\{s}", 10) +
      pad(fmt2(t_naive), 12) +
      pad(fmt2(t_flash), 12) +
      pad(fmt2(t_naive / t_flash) + "x", 10) +
      pad(gflops(t_naive, 1, 1, s, s, 64, 64), 13) +
      pad(gflops(t_flash, 1, 1, s, s, 64, 64), 13) +
      fmt_err(diff),
    )
  }
}

///|
/// Head-dim sweep at fixed sequence length (non-causal).
fn sweep_headdim(reps : Int) -> Unit {
  println("")
  println("## 3. Head dim sweep (n = m = 1024, d = dv, non-causal)")
  println("")
  println(
    pad("d", 8) +
    pad("naive us", 12) +
    pad("flash us", 12) +
    pad("naive GF/s", 13) +
    pad("flash GF/s", 13) +
    "max diff",
  )
  let cfg = @flash.AttnConfig::new(heads=1)
  for d in [32, 64, 128] {
    let (q, k, v) = make_qkv(1, 1, 1, 1024, 1024, d, d)
    let t_naive = timed_us(
      fn() { @flash.naive_attention(q, k, v, cfg) |> ignore },
      reps,
    )
    let t_flash = timed_us(
      fn() { @flash.flash_attention(q, k, v, cfg) |> ignore },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q, k, v, cfg),
      @flash.flash_attention(q, k, v, cfg),
    )
    println(
      pad("\{d}", 8) +
      pad(fmt2(t_naive), 12) +
      pad(fmt2(t_flash), 12) +
      pad(gflops(t_naive, 1, 1, 1024, 1024, d, d), 13) +
      pad(gflops(t_flash, 1, 1, 1024, 1024, d, d), 13) +
      fmt_err(diff),
    )
  }
}

///|
/// Tile-size sweep for flash at fixed problem size (non-causal).
fn sweep_blocksize(reps : Int) -> Unit {
  println("")
  println("## 4. Flash tile size sweep (n = m = 2048, d = dv = 64, non-causal)")
  println("")
  println(
    pad("tiles", 12) + pad("flash us", 12) + pad("flash GF/s", 13) + "max diff",
  )
  let cfg = @flash.AttnConfig::new(heads=1)
  for b in [16, 32, 64, 128, 256] {
    let (q, k, v) = make_qkv(1, 1, 1, 2048, 2048, 64, 64)
    let t_flash = timed_us(
      fn() {
        @flash.flash_attention(q, k, v, cfg, block_rows=b, block_cols=b)
        |> ignore
      },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q, k, v, cfg),
      @flash.flash_attention(q, k, v, cfg, block_rows=b, block_cols=b),
    )
    println(
      pad("\{b}x\{b}", 12) +
      pad(fmt2(t_flash), 12) +
      pad(gflops(t_flash, 1, 1, 2048, 2048, 64, 64), 13) +
      fmt_err(diff),
    )
  }
}

///|
fn main {
  let reps = 3
  println("# flashmoon vs naive attention benchmark (min of \{reps} runs)")
  println("")
  sweep_scenarios(reps)
  sweep_seqlen(reps)
  sweep_headdim(reps)
  sweep_blocksize(reps)
}
