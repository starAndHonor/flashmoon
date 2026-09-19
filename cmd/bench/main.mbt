///|
/// FlashAttention evaluation harness.
///
/// Reports wall-clock (min of reps, wasm release), throughput in GFLOP/s,
/// and numerical agreement between the naive reference and the flash kernel.
///
/// Run: moon run cmd/bench --target wasm --release

///|
/// Deterministic pseudo-random value in [-1, 1).
fn rand_val(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
/// q [b,h,n,d], k/v [b,h,m,d].
fn make_qkv(
  b : Int,
  h : Int,
  n : Int,
  m : Int,
  d : Int,
) -> (@nparray.NpArray, @nparray.NpArray, @nparray.NpArray) {
  let mk = fn(rows : Int, cols : Int, k : Int) -> @nparray.NpArray {
    let data = FixedArray::make(b * h * rows * cols, 0.0F)
    for i in 0..<(b * h * rows) {
      for j in 0..<cols {
        data.unsafe_set(i * cols + j, rand_val(i, j, k))
      }
    }
    @tensor.Tensor::from_host(data, [b, h, rows, cols])
  }
  (mk(n, d, 0), mk(m, d, 1), mk(m, d, 2))
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
/// Total attention FLOPs per slice: QK^T (2*n*m*d) + PV (2*n*m*dv), dv = d.
fn flops(b : Int, h : Int, n : Int, m : Int, d : Int) -> Double {
  4.0 *
  b.to_double() *
  h.to_double() *
  n.to_double() *
  m.to_double() *
  d.to_double()
}

///|
fn gflops(us : Double, b : Int, h : Int, n : Int, m : Int, d : Int) -> String {
  fmt2(flops(b, h, n, m, d) / us / 1000.0)
}

///|
/// Row: sequence-length sweep, naive vs flash (default 64x64 tiles).
fn sweep_seqlen(reps : Int) -> Unit {
  println("## 1. Sequence length sweep (n = m, d = dv = 64, flash tiles 64x64)")
  println("")
  println(
    pad("size", 10) +
    pad("naive ms", 12) +
    pad("flash ms", 12) +
    pad("speedup", 10) +
    pad("naive GF/s", 13) +
    pad("flash GF/s", 13) +
    "max diff",
  )
  let cfg = @flash.AttnConfig::new(heads=1)
  for s in [128, 256, 512, 1024, 2048, 4096] {
    let (q, k, v) = make_qkv(1, 1, s, s, 64)
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
      pad(fmt2(t_naive / 1000.0), 12) +
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(fmt2(t_naive / t_flash) + "x", 10) +
      pad(gflops(t_naive, 1, 1, s, s, 64), 13) +
      pad(gflops(t_flash, 1, 1, s, s, 64), 13) +
      fmt_err(diff),
    )
  }
}

///|
/// Row: head-dim sweep at fixed sequence length.
fn sweep_headdim(reps : Int) -> Unit {
  println("")
  println("## 2. Head dim sweep (n = m = 1024, d = dv)")
  println("")
  println(
    pad("d", 8) +
    pad("naive ms", 12) +
    pad("flash ms", 12) +
    pad("naive GF/s", 13) +
    pad("flash GF/s", 13) +
    "max diff",
  )
  let cfg = @flash.AttnConfig::new(heads=1)
  for d in [32, 64, 128] {
    let (q, k, v) = make_qkv(1, 1, 1024, 1024, d)
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
      pad(fmt2(t_naive / 1000.0), 12) +
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(gflops(t_naive, 1, 1, 1024, 1024, d), 13) +
      pad(gflops(t_flash, 1, 1, 1024, 1024, d), 13) +
      fmt_err(diff),
    )
  }
}

///|
/// Row: tile-size sweep for flash at fixed problem size.
fn sweep_blocksize(reps : Int) -> Unit {
  println("")
  println("## 3. Flash tile size sweep (n = m = 2048, d = dv = 64)")
  println("")
  println(
    pad("tiles", 12) + pad("flash ms", 12) + pad("flash GF/s", 13) + "max diff",
  )
  let cfg = @flash.AttnConfig::new(heads=1)
  for b in [16, 32, 64, 128, 256] {
    let (q, k, v) = make_qkv(1, 1, 2048, 2048, 64)
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
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(gflops(t_flash, 1, 1, 2048, 2048, 64), 13) +
      fmt_err(diff),
    )
  }
}

///|
/// Row: batched multi-head causal sweep (GQA included).
fn sweep_mha(reps : Int) -> Unit {
  println("")
  println("## 4. Multi-head causal sweep (n = m = 512, d = dv = 64)")
  println("")
  println(
    pad("shape", 18) +
    pad("naive ms", 12) +
    pad("flash ms", 12) +
    pad("flash GF/s", 13) +
    "max diff",
  )
  for shape in [(1, 8, 8), (2, 8, 8), (2, 8, 2)] {
    let (b, h, hkv) = shape
    let (q, k, v) = make_qkv(b, h, 512, 512, 64)
    // note: make_qkv gives k/v h heads; for GQA reuse hkv via slice dims
    let (q2, k2, v2) = if hkv == h {
      (q, k, v)
    } else {
      make_qkv(b, hkv, 512, 512, 64) |> fn(kv) { (q, kv.1, kv.2) }
    }
    let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal=true)
    let t_naive = timed_us(
      fn() { @flash.naive_attention(q2, k2, v2, cfg) |> ignore },
      reps,
    )
    let t_flash = timed_us(
      fn() { @flash.flash_attention(q2, k2, v2, cfg) |> ignore },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q2, k2, v2, cfg),
      @flash.flash_attention(q2, k2, v2, cfg),
    )
    println(
      pad("b\{b} h\{h} hkv\{hkv}", 18) +
      pad(fmt2(t_naive / 1000.0), 12) +
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(gflops(t_flash, b, h, 512, 512, 64), 13) +
      fmt_err(diff),
    )
  }
}

///|
fn main {
  let reps = 3
  println("# FlashAttention benchmark (wasm, release, min of \{reps} runs)")
  println("")
  sweep_seqlen(reps)
  sweep_headdim(reps)
  sweep_blocksize(reps)
  sweep_mha(reps)
}
