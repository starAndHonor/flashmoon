///|
/// WebGPU kernel correctness + benchmark host (JS target).
/// Runs under Deno (scripts/webgpu_host.js) or a browser; results are
/// posted via fetch("/result").

///|
extern "js" fn post(s : String) -> Unit =
  #|(s) => { fetch('/result', { method: 'POST', body: s }); }

///|
extern "js" fn now_ms() -> Double =
  #|() => performance.now()

///|
fn rand_val(i : Int, j : Int, k : Int) -> Float {
  (Float::from_int((i * 37 + j * 91 + k * 13 + 11) % 29) - 14.0F) / 14.0F
}

///|
fn make_mat(rows : Int, cols : Int, k : Int) -> FixedArray[Float] {
  let data = FixedArray::make(rows * cols, 0.0F)
  for i in 0..<rows {
    for j in 0..<cols {
      data.unsafe_set(i * cols + j, rand_val(i, j, k))
    }
  }
  data
}

///|
fn as_tensor(
  data : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> @nparray.NpArray {
  @tensor.Tensor::from_host(data, [1, 1, rows, cols])
}

///|
fn max_diff(a : FixedArray[Float], b : FixedArray[Float]) -> Float {
  let mut md = 0.0F
  for i in 0..<a.length() {
    let d = (a.unsafe_get(i) - b.unsafe_get(i)).abs()
    if d > md {
      md = d
    }
  }
  md
}

///|
/// bf16 truncation, matching the GPU kernels' exact upcast: keeps the
/// CPU oracle lossless against bf16 weight buffers.
fn bf16r(x : Float) -> Float {
  Float::reinterpret_from_int(x.reinterpret_as_int() & -65536)
}

///|
/// CPU reference: y[out] = W[out,in] @ x[in]
fn cpu_matvec(
  w : FixedArray[Float],
  x : FixedArray[Float],
  out : Int,
  inn : Int,
) -> FixedArray[Float] {
  let y = FixedArray::make(out, 0.0F)
  for o in 0..<out {
    let base = o * inn
    let mut acc = 0.0F
    for i in 0..<inn {
      acc += w.unsafe_get(base + i) * x.unsafe_get(i)
    }
    y.unsafe_set(o, acc)
  }
  y
}

///|
/// CPU reference: Y[s,out] = X[s,in] @ W[out,in]^T
fn cpu_matmul(
  x : FixedArray[Float],
  w : FixedArray[Float],
  s : Int,
  out : Int,
  inn : Int,
) -> FixedArray[Float] {
  let y = FixedArray::make(s * out, 0.0F)
  for r in 0..<s {
    for o in 0..<out {
      let mut acc = 0.0F
      for i in 0..<inn {
        acc += x.unsafe_get(r * inn + i) * w.unsafe_get(o * inn + i)
      }
      y.unsafe_set(r * out + o, acc)
    }
  }
  y
}

///|
fn check_attention(g : @gpu.Gpu, next : (Unit) -> Unit) -> Unit {
  let (q, k, v) = (
    make_mat(128, 64, 0),
    make_mat(128, 64, 1),
    make_mat(128, 64, 2),
  )
  let qb = g.upload(q)
  let kb = g.upload(k)
  let vb = g.upload(v)
  let outb = g.alloc(128 * 64)
  g.flash_attention(qb, kb, vb, outb, 128, 128, 64, 64)
  g.readback(outb, 128 * 64, fn(got) {
    let want = @flash.naive_attention(
      as_tensor(q, 128, 64),
      as_tensor(k, 128, 64),
      as_tensor(v, 128, 64),
      @flash.AttnConfig::new(heads=1),
    )
    post("attention 128x128 d=64: max_diff=\{max_diff(got, want.value().data)}")
    let (q2, k2, v2) = (
      make_mat(4096, 64, 0),
      make_mat(4096, 64, 1),
      make_mat(4096, 64, 2),
    )
    let qb2 = g.upload(q2)
    let kb2 = g.upload(k2)
    let vb2 = g.upload(v2)
    let outb2 = g.alloc(4096 * 64)
    g.flash_attention(qb2, kb2, vb2, outb2, 4096, 4096, 64, 64) // warm (compile)
    g.sync(fn(_) {
      let t0 = now_ms()
      g.flash_attention(qb2, kb2, vb2, outb2, 4096, 4096, 64, 64)
      g.sync(fn(_) {
        let dt = now_ms() - t0
        let gflops = 4.0 * 4096.0 * 4096.0 * 64.0 / (dt * 1000.0) / 1000.0
        post(
          "attention 4096x4096 d=64: \{dt} ms, \{gflops} GFLOP/s (dispatch only)",
        )
        next(())
      })
    })
  })
}

///|
fn check_gemv(g : @gpu.Gpu, next : (Unit) -> Unit) -> Unit {
  // Correctness: 512 x 256
  let w = make_mat(512, 256, 3).map(bf16r)
  let x = make_mat(256, 1, 4)
  let wb = g.upload_raw_buf(@gpu.to_bf16(w))
  let xb = g.upload(x)
  let yb = g.alloc(512)
  g.matvec(wb, xb, yb, 512, 256)
  g.readback(yb, 512, fn(got) {
    post("gemv 512x256: max_diff=\{max_diff(got, cpu_matvec(w, x, 512, 256))}")
    // Perf: lm_head decode shape 151936 x 1024. Weights GPU-allocated
    // garbage: 155M floats exceed V8's max JS-array length (2^27), and
    // values don't affect kernel speed.
    let wb2 = g.alloc(151936 * 1024 / 2)
    let x2 = make_mat(1024, 1, 6)
    let xb2 = g.upload(x2)
    let yb2 = g.alloc(151936)
    g.matvec(wb2, xb2, yb2, 151936, 1024) // warm
    g.sync(fn(_) {
      let t0 = now_ms()
      for _ in 0..<10 {
        g.matvec(wb2, xb2, yb2, 151936, 1024)
      }
      g.sync(fn(_) {
        let dt = (now_ms() - t0) / 10.0
        let gb = 151936.0 * 1024.0 * 2.0 / 1.0e9
        post(
          "gemv 151936x1024: \{dt} ms, \{gb / (dt / 1000.0)} GB/s (dispatch only)",
        )
        next(())
      })
    })
  })
}

///|
fn check_gemm(g : @gpu.Gpu, next : (Unit) -> Unit) -> Unit {
  // Correctness: s=32, out=256, in=512
  let x = make_mat(32, 512, 7)
  let w = make_mat(256, 512, 8).map(bf16r)
  let xb = g.upload(x)
  let wb = g.upload_raw_buf(@gpu.to_bf16(w))
  let yb = g.alloc(32 * 256)
  g.matmul(xb, wb, yb, 32, 256, 512)
  g.readback(yb, 32 * 256, fn(got) {
    post(
      "gemm 32x256x512: max_diff=\{max_diff(got, cpu_matmul(x, w, 32, 256, 512))}",
    )
    // Perf: prefill shape s=128, q_proj 2048x1024
    let x2 = make_mat(128, 1024, 9)
    let w2 = make_mat(2048, 1024, 10).map(bf16r)
    let xb2 = g.upload(x2)
    let wb2 = g.upload_raw_buf(@gpu.to_bf16(w2))
    let yb2 = g.alloc(128 * 2048)
    g.matmul(xb2, wb2, yb2, 128, 2048, 1024) // warm
    g.sync(fn(_) {
      let t0 = now_ms()
      g.matmul(xb2, wb2, yb2, 128, 2048, 1024)
      g.sync(fn(_) {
        let dt = now_ms() - t0
        let gflops = 2.0 * 128.0 * 2048.0 * 1024.0 / (dt * 1000.0) / 1000.0
        post("gemm 128x2048x1024: \{dt} ms, \{gflops} GFLOP/s (dispatch only)")
        next(())
      })
    })
  })
}

///|
/// Fill a 4D [b,h,rows,cols] tensor with the deterministic pattern.
fn make_mat4(
  b : Int,
  h : Int,
  rows : Int,
  cols : Int,
  k : Int,
) -> @nparray.NpArray {
  let data = FixedArray::make(b * h * rows * cols, 0.0F)
  for i in 0..<(b * h * rows) {
    for j in 0..<cols {
      data[i * cols + j] = rand_val(i, j, k)
    }
  }
  @tensor.Tensor::from_host(data, [b, h, rows, cols])
}

///|
/// Property checks for the WebGPU 4D flash backend vs the CPU naive
/// oracle: MHA/GQA/MQA, causal, cross-attention with cache, odd dims.
fn check_flash4d(g : @gpu.Gpu, next : (Unit) -> Unit) -> Unit {
  let cases : Array[(Int, Int, Int, Int, Int, Int, Int, Bool)] = [
    (1, 1, 1, 128, 128, 64, 64, false),
    (2, 4, 2, 64, 64, 32, 32, true),
    (1, 3, 1, 16, 64, 24, 40, true),
    (2, 2, 2, 48, 96, 13, 7, false),
  ]
  fn run(idx : Int) -> Unit {
    if idx >= cases.length() {
      next(())
      return
    }
    let (b, h, hkv, sq, skv, d, dv, causal) = cases[idx]
    let q = make_mat4(b, h, sq, d, 0)
    let k = make_mat4(b, hkv, skv, d, 1)
    let v = make_mat4(b, hkv, skv, dv, 2)
    let cfg = @flash.AttnConfig::new(heads=h, kv_heads=hkv, causal~)
    @flashgpu.flash_attention(g, q, k, v, cfg, fn(got) {
      let want = @flash.naive_attention(q, k, v, cfg)
      let diff = max_diff(got.value().data, want.value().data)
      post(
        "flash4d b\{b} h\{h}/\{hkv} \{sq}x\{skv} d\{d}/\{dv} causal=\{causal}: max_diff=\{diff}",
      )
      run(idx + 1)
    })
  }
  run(0)
}

///|
fn main {
  post("gpu ops demo start")
  @gpu.Gpu::init(
    fn(g) {
      check_attention(g, fn(_) {
        check_flash4d(g, fn(_) {
          check_gemv(g, fn(_) { check_gemm(g, fn(_) { post("done") }) })
        })
      })
    },
    log=post,
  )
}
