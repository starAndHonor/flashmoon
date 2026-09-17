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
fn make_qkv(
  n : Int,
  m : Int,
  d : Int,
) -> (@nparray.NpArray, @nparray.NpArray, @nparray.NpArray) {
  let mk = fn(rows : Int, cols : Int, k : Int) -> @nparray.NpArray {
    let data = FixedArray::make(rows * cols, 0.0F)
    for i in 0..<rows {
      for j in 0..<cols {
        data.unsafe_set(i * cols + j, rand_val(i, j, k))
      }
    }
    @tensor.Tensor::from_host(data, [rows, cols])
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
/// Total attention FLOPs: QK^T (2*n*m*d) + PV (2*n*m*dv), with dv = d here.
fn flops(n : Int, m : Int, d : Int) -> Double {
  4.0 * n.to_double() * m.to_double() * d.to_double()
}

///|
fn gflops(us : Double, n : Int, m : Int, d : Int) -> String {
  fmt2(flops(n, m, d) / us / 1000.0)
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
  for s in [128, 256, 512, 1024, 2048, 4096] {
    let (q, k, v) = make_qkv(s, s, 64)
    let t_naive = timed_us(
      fn() { @flash.naive_attention(q, k, v) |> ignore },
      reps,
    )
    let t_flash = timed_us(
      fn() { @flash.flash_attention(q, k, v) |> ignore },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q, k, v),
      @flash.flash_attention(q, k, v),
    )
    println(
      pad("\{s}x\{s}", 10) +
      pad(fmt2(t_naive / 1000.0), 12) +
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(fmt2(t_naive / t_flash) + "x", 10) +
      pad(gflops(t_naive, s, s, 64), 13) +
      pad(gflops(t_flash, s, s, 64), 13) +
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
  for d in [32, 64, 128] {
    let (q, k, v) = make_qkv(1024, 1024, d)
    let t_naive = timed_us(
      fn() { @flash.naive_attention(q, k, v) |> ignore },
      reps,
    )
    let t_flash = timed_us(
      fn() { @flash.flash_attention(q, k, v) |> ignore },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q, k, v),
      @flash.flash_attention(q, k, v),
    )
    println(
      pad("\{d}", 8) +
      pad(fmt2(t_naive / 1000.0), 12) +
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(gflops(t_naive, 1024, 1024, d), 13) +
      pad(gflops(t_flash, 1024, 1024, d), 13) +
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
  for b in [16, 32, 64, 128, 256] {
    let (q, k, v) = make_qkv(2048, 2048, 64)
    let t_flash = timed_us(
      fn() {
        @flash.flash_attention(q, k, v, block_rows=b, block_cols=b) |> ignore
      },
      reps,
    )
    let diff = max_abs_diff(
      @flash.naive_attention(q, k, v),
      @flash.flash_attention(q, k, v, block_rows=b, block_cols=b),
    )
    println(
      pad("\{b}x\{b}", 12) +
      pad(fmt2(t_flash / 1000.0), 12) +
      pad(gflops(t_flash, 2048, 2048, 64), 13) +
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
}
