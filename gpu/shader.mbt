///|
/// WGSL register-array bound; must be >= any dv passed to attention_shader.
const MAX_DV_WGSL : Int = 128

///|
/// K/V tile rows per workgroup block. K tile + V tile must fit workgroup
/// shared memory (32 * 64 * 4 * 2 = 16KB at D = DV = 64).
const BC_WGSL : Int = 16

///|
/// Emit `dv` textually unrolled statements with constant indices,
/// so the WGSL compiler keeps `o` in registers instead of local memory.
fn unroll_dv(dv : Int, body : (StringBuilder, Int) -> Unit) -> String {
  let buf = StringBuilder::new()
  for c in 0..<dv {
    body(buf, c)
    buf.write_char('\n')
  }
  buf.to_string()
}

///|
/// WGSL source for the flash-attention forward kernel.
///
/// One workgroup per 64 query rows (one thread per row). Threads
/// cooperatively stage K/V tiles in workgroup shared memory, then each
/// thread streams the tile with an online softmax update (running max /
/// running sum / rescaled output accumulator). The full [n, m] score
/// matrix is never materialized.
/// Dimensions are baked into the shader as constants; accumulator loops
/// and the q row are textually unrolled to stay in registers.
pub fn attention_shader(n : Int, m : Int, d : Int, dv : Int) -> String {
  let accum = unroll_dv(dv, fn(buf, c) {
    buf.write_string("o[\{c}] = o[\{c}] * alpha + p * vs[j * DV + \{c}u];")
  })
  let qload = unroll_dv(d, fn(buf, t) {
    buf.write_string("qr[\{t}] = q[i * D + \{t}u];")
  })
  let dot = unroll_dv(d, fn(buf, t) {
    buf.write_string("s = s + qr[\{t}] * ks[j * D + \{t}u];")
  })
  let store = unroll_dv(dv, fn(buf, c) {
    buf.write_string("outp[i * DV + \{c}u] = o[\{c}] * inv;")
  })
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> k : array<f32>;
    $|@group(0) @binding(2) var<storage, read> v : array<f32>;
    $|@group(0) @binding(3) var<storage, read_write> outp : array<f32>;
    $|
    $|const N : u32 = \{n}u;
    $|const M : u32 = \{m}u;
    $|const D : u32 = \{d}u;
    $|const DV : u32 = \{dv}u;
    $|const MAX_DV : u32 = \{MAX_DV_WGSL}u;
    $|const BC : u32 = \{BC_WGSL}u;
    $|const WG : u32 = 64u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|var<workgroup> ks : array<f32, BC * D>;
    $|var<workgroup> vs : array<f32, BC * DV>;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(global_invocation_id) gid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let i = gid.x;
    $|  let scale = 1.0 / sqrt(f32(D));
    $|  var o : array<f32, MAX_DV>;
    $|  var qr : array<f32, MAX_DV>;
    $|  var m_run = NEG_BIG;
    $|  var l_run = 0.0;
    $|  if (i < N) {
    $|    \{qload}
    $|  }
    $|  for (var j0 = 0u; j0 < M; j0 = j0 + BC) {
    $|    // cooperative tile load: threads share K/V global reads
    $|    for (var t = lid.x; t < BC * D; t = t + WG) {
    $|      let r = t / D;
    $|      let c = t % D;
    $|      let gj = j0 + r;
    $|      ks[t] = select(0.0, k[gj * D + c], gj < M);
    $|    }
    $|    for (var t = lid.x; t < BC * DV; t = t + WG) {
    $|      let r = t / DV;
    $|      let c = t % DV;
    $|      let gj = j0 + r;
    $|      vs[t] = select(0.0, v[gj * DV + c], gj < M);
    $|    }
    $|    workgroupBarrier();
    $|    if (i < N) {
    $|      for (var j = 0u; j < BC; j = j + 1u) {
    $|        if (j0 + j >= M) { break; }
    $|        var s = 0.0;
    $|        \{dot}
    $|        s = s * scale;
    $|        let m_new = max(m_run, s);
    $|        let alpha = exp(m_run - m_new);
    $|        let p = exp(s - m_new);
    $|        l_run = l_run * alpha + p;
    $|        \{accum}
    $|        m_run = m_new;
    $|      }
    $|    }
    $|    workgroupBarrier();
    $|  }
    $|  if (i < N) {
    $|    let inv = 1.0 / l_run;
    $|    \{store}
    $|  }
    $|}
  src
}
