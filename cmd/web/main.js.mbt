///|
/// WebGPU flash attention demo + benchmark host (JS target, browser).
/// Posts results back to the harness server via POST /result.

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
fn main {
  post("webgpu demo start")
  // Correctness: 128x128 d=64 vs CPU naive
  let (q, k, v) = make_qkv(128, 128, 64)
  @gpu.flash_attention_gpu(q, k, v, fn(gpu_out) {
    let cpu_out = @flashmoon.naive_attention(q, k, v)
    let da = gpu_out.value().data
    let db = cpu_out.value().data
    let mut max_diff = 0.0F
    for i in 0..<da.length() {
      let diff = (da.unsafe_get(i) - db.unsafe_get(i)).abs()
      if diff > max_diff {
        max_diff = diff
      }
    }
    post("correctness 128x128 d=64: max_diff=\{max_diff}")
    // Performance: 4096x4096 d=64, dispatch+readback wall time
    let (qb, kb, vb) = make_qkv(4096, 4096, 64)
    let t0 = now_ms()
    @gpu.flash_attention_gpu(qb, kb, vb, log=post, fn(out) {
      let t1 = now_ms()
      ignore(out)
      let flops = 4.0 * 4096.0 * 4096.0 * 64.0
      let gflops = flops / ((t1 - t0) * 1000.0) / 1000.0
      post(
        "perf 4096x4096 d=64: \{t1 - t0} ms, \{gflops} GFLOP/s (incl. upload+readback)",
      )
      post("done")
    })
  })
}
