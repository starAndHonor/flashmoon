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
/// Emit a 4-way partially-summed dot product over `t in 0..<d`:
/// four independent accumulator chains (ILP) instead of one serial
/// dependency chain into a single scalar.
fn dot4(d : Int, qref : String, kref : String) -> String {
  let buf = StringBuilder::new()
  buf.write_string("var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;")
  buf.write_char('\n')
  for t in 0..<d {
    buf.write_string("s\{t % 4} = s\{t % 4} + \{qref}[\{t}] * \{kref}\{t}u];")
    buf.write_char('\n')
  }
  buf.write_string("let s_dot = (s0 + s1) + (s2 + s3);")
  buf.write_char('\n')
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
  let dot = dot4(d, "qr", "ks[j * D + ")
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
    $|        \{dot}
    $|        let s = s_dot * scale;
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

///|
/// Kernel 1 of the split-K flash attention (FlashDecoding-style), vec4.
///
/// One workgroup of 64 threads per (row-block of 64 query rows, key split
/// s): thread-per-row, K/V staged in shared vec4 tiles (global K/V traffic
/// /64, 16B-aligned coalesced loads), keys split S-ways for occupancy.
/// All hot loops operate on vec4: 1 load + 2 vector ops per 4 columns.
/// The q row is broadcast from shared memory (register budget goes to the
/// 32-vec4 output accumulator). Each thread writes its own partial record
/// (m, l, o[dv]) as vec4s; no cross-thread merge.
/// Dispatch: (splits, ceil(n/64), 1). Requires d, dv <= 64, both % 4 == 0.
pub fn partial_shader(
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
  splits : Int,
) -> String {
  let chunk = (m + splits - 1) / splits
  let d4 = d / 4
  let dv4 = dv / 4
  let qdecl = unroll_dv(d4, fn(buf, t) {
    buf.write_string("var qr\{t} : vec4<f32>;")
  })
  let qload = unroll_dv(d4, fn(buf, t) {
    buf.write_string("qr\{t} = q[i * D4 + \{t}u];")
  })
  let odecl = unroll_dv(dv4, fn(buf, c) {
    buf.write_string("var o\{c} = vec4<f32>(0.0);")
  })
  let dot = unroll_dv(d4, fn(buf, t) {
    buf.write_string("s4 = s4 + qr\{t} * ks[j * D4 + \{t}u];")
  })
  let accum = unroll_dv(dv4, fn(buf, c) {
    buf.write_string("o\{c} = o\{c} * alpha4 + p4 * vs[j * DV4 + \{c}u];")
  })
  let pwrite = unroll_dv(dv4, fn(buf, c) {
    buf.write_string("part[rb + 1u + \{c}u] = o\{c};")
  })
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> k : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read> v : array<vec4<f32>>;
    $|@group(0) @binding(3) var<storage, read_write> part : array<vec4<f32>>;
    $|
    $|const N : u32 = \{n}u;
    $|const M : u32 = \{m}u;
    $|const D4 : u32 = \{d4}u;
    $|const DV4 : u32 = \{dv4}u;
    $|const S : u32 = \{splits}u;
    $|const CHUNK : u32 = \{chunk}u;
    $|const BC : u32 = \{BC_WGSL}u;
    $|const WG : u32 = 64u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|var<workgroup> ks : array<vec4<f32>, BC * 16>;
    $|var<workgroup> vs : array<vec4<f32>, BC * 16>;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let sp = wid.x;
    $|  let i = wid.y * WG + lid.x;
    $|  let j_begin = sp * CHUNK;
    $|  let j_end = min(j_begin + CHUNK, M);
    $|  let scale = 1.0 / sqrt(f32(D4 * 4u));
    $|  \{odecl}
    $|  \{qdecl}
    $|  var m_run = NEG_BIG;
    $|  var l_run = 0.0;
    $|  if (i < N) {
    $|    \{qload}
    $|  }
    $|  for (var j0 = j_begin; j0 < j_end; j0 = j0 + BC) {
    $|    for (var t = lid.x; t < BC * D4; t = t + WG) {
    $|      let r = t / D4;
    $|      let c = t % D4;
    $|      let gj = j0 + r;
    $|      ks[t] = select(vec4<f32>(0.0), k[gj * D4 + c], gj < j_end);
    $|    }
    $|    for (var t = lid.x; t < BC * DV4; t = t + WG) {
    $|      let r = t / DV4;
    $|      let c = t % DV4;
    $|      let gj = j0 + r;
    $|      vs[t] = select(vec4<f32>(0.0), v[gj * DV4 + c], gj < j_end);
    $|    }
    $|    workgroupBarrier();
    $|    if (i < N) {
    $|      for (var j = 0u; j < BC; j = j + 1u) {
    $|        if (j0 + j >= j_end) { break; }
    $|        var s4 = vec4<f32>(0.0);
    $|        \{dot}
    $|        let s = (s4.x + s4.y + s4.z + s4.w) * scale;
    $|        let m_new = max(m_run, s);
    $|        let alpha = exp(m_run - m_new);
    $|        let p = exp(s - m_new);
    $|        let alpha4 = vec4<f32>(alpha);
    $|        let p4 = vec4<f32>(p);
    $|        l_run = l_run * alpha + p;
    $|        \{accum}
    $|        m_run = m_new;
    $|      }
    $|    }
    $|    workgroupBarrier();
    $|  }
    $|  if (i < N) {
    $|    let rb = (i * S + sp) * (DV4 + 1u);
    $|    part[rb] = vec4<f32>(m_run, l_run, 0.0, 0.0);
    $|    \{pwrite}
    $|  }
    $|}
  src
}

///|
/// Kernel 2: combine the per-split partials into the final output, vec4.
///
/// One thread per output vec4 (i, c4); rescales and sums the S partial
/// records for its row. Dispatch: ceil(n*dv4/256) workgroups.
pub fn combine_shader(n : Int, dv : Int, splits : Int) -> String {
  let dv4 = dv / 4
  let src =
    $|@group(0) @binding(0) var<storage, read> part : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read_write> outp : array<vec4<f32>>;
    $|
    $|const N : u32 = \{n}u;
    $|const DV4 : u32 = \{dv4}u;
    $|const S : u32 = \{splits}u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $|  let flat = gid.x;
    $|  if (flat >= N * DV4) { return; }
    $|  let i = flat / DV4;
    $|  let c = flat % DV4;
    $|  let stride = DV4 + 1u;
    $|  let base0 = i * S * stride;
    $|  var m = NEG_BIG;
    $|  for (var s = 0u; s < S; s = s + 1u) {
    $|    m = max(m, part[base0 + s * stride].x);
    $|  }
    $|  var l = 0.0;
    $|  var acc = vec4<f32>(0.0);
    $|  for (var s = 0u; s < S; s = s + 1u) {
    $|    let b = base0 + s * stride;
    $|    let e = exp(part[b].x - m);
    $|    l = l + part[b].y * e;
    $|    acc = acc + part[b + 1u + c] * e;
    $|  }
    $|  outp[flat] = acc / vec4<f32>(l);
    $|}
  src
}

///|
/// GEMV kernel: y[out] = W[out, in] @ x[in], W row-major.
///
/// 256-thread workgroup covers 8 output rows (one 32-lane warp per row):
/// x staged in shared as vec4 (<= 12KB for in <= 3072), lanes stream their
/// row with coalesced 512B-per-instruction vec4 reads, then a per-warp
/// shared reduction (8 rows reduced in parallel). Grid:
/// (ceil(out/8), 1). Requires in % 4 == 0.
pub fn gemv_shader(out : Int, inn : Int) -> String {
  let in4 = inn / 4
  let src =
    $|@group(0) @binding(0) var<storage, read> w : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> x : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read_write> y : array<f32>;
    $|
    $|const OUT : u32 = \{out}u;
    $|const IN4 : u32 = \{in4}u;
    $|
    $|var<workgroup> xs : array<vec4<f32>, 768>;
    $|var<workgroup> part : array<f32, 256>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let rg = lid.x / 32u; // row group (warp) 0..7
    $|  let lane = lid.x % 32u;
    $|  let o = wid.x * 8u + rg;
    $|  let oo = min(o, OUT - 1u);
    $|  for (var t = lid.x; t < IN4; t = t + 256u) {
    $|    xs[t] = x[t];
    $|  }
    $|  workgroupBarrier();
    $|  var acc = 0.0;
    $|  for (var t = lane; t < IN4; t = t + 32u) {
    $|    acc = acc + dot(w[oo * IN4 + t], xs[t]);
    $|  }
    $|  part[lid.x] = acc;
    $|  workgroupBarrier();
    $|  if (lane == 0u && o < OUT) {
    $|    var sum = 0.0;
    $|    for (var t = 0u; t < 32u; t = t + 1u) {
    $|      sum = sum + part[rg * 32u + t];
    $|    }
    $|    y[o] = sum;
    $|  }
    $|}
  src
}

///|
/// GEMM kernel: Y[s, out] = X[s, in] @ W[out, in]^T, both row-major.
///
/// Classic 16x16 output tile per 256-thread workgroup; k streamed in
/// 4-vec4 chunks through shared memory (2KB tiles), each thread owns one
/// output element and accumulates with vec4 dots. Edges zero-padded via
/// select. Requires in % 4 == 0.
pub fn matmul_shader(s : Int, out : Int, inn : Int) -> String {
  let in4 = inn / 4
  let src =
    $|@group(0) @binding(0) var<storage, read> x : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> w : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read_write> y : array<f32>;
    $|
    $|const S : u32 = \{s}u;
    $|const OUT : u32 = \{out}u;
    $|const IN4 : u32 = \{in4}u;
    $|
    $|var<workgroup> xs : array<vec4<f32>, 64>;
    $|var<workgroup> ws : array<vec4<f32>, 64>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let col0 = wid.x * 16u;
    $|  let row0 = wid.y * 16u;
    $|  let tx = lid.x % 16u;
    $|  let ty = lid.x / 16u;
    $|  var acc = 0.0;
    $|  for (var k0 = 0u; k0 < IN4; k0 = k0 + 4u) {
    $|    for (var t = lid.x; t < 64u; t = t + 256u) {
    $|      let r = t / 4u;
    $|      let c = t % 4u;
    $|      let gi = row0 + r;
    $|      let kk = k0 + c;
    $|      xs[t] = select(vec4<f32>(0.0), x[gi * IN4 + kk], gi < S && kk < IN4);
    $|      let gj = col0 + r;
    $|      ws[t] = select(vec4<f32>(0.0), w[gj * IN4 + kk], gj < OUT && kk < IN4);
    $|    }
    $|    workgroupBarrier();
    $|    for (var kk = 0u; kk < 4u; kk = kk + 1u) {
    $|      acc = acc + dot(xs[ty * 4u + kk], ws[tx * 4u + kk]);
    $|    }
    $|    workgroupBarrier();
    $|  }
    $|  let gi = row0 + ty;
    $|  let gj = col0 + tx;
    $|  if (gi < S && gj < OUT) {
    $|    y[gi * OUT + gj] = acc;
    $|  }
    $|}
  src
}

///|
/// RMSNorm: y[r, i] = x[r, i] * rsqrt(mean(x[r,:]^2) + eps) * w[i].
/// One 256-thread workgroup per row, vec4 loads, shared-memory reduction.
/// In-place safe (x == y): each thread writes only indices it read.
/// Grid: (rows, 1). Requires n % 4 == 0, n/4 <= 768 * 4? No shared staging
/// of x — n is only bounded by the reduction loop, any n % 4 == 0 works.
pub fn rmsnorm_shader(n : Int, eps : Float) -> String {
  let n4 = n / 4
  let src =
    $|// x is read_write so in-place calls (y == x) bind the same buffer
    $|// under one storage usage (read-only + read-write would be rejected).
    $|@group(0) @binding(0) var<storage, read_write> x : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> w : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read_write> y : array<vec4<f32>>;
    $|
    $|const N : u32 = \{n}u;
    $|const N4 : u32 = \{n4}u;
    $|const EPS : f32 = \{eps};
    $|
    $|var<workgroup> part : array<f32, 256>;
    $|var<workgroup> inv_s : f32;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let base = wid.x * N4;
    $|  var ss = 0.0;
    $|  for (var t = lid.x; t < N4; t = t + 256u) {
    $|    let v = x[base + t];
    $|    ss = ss + dot(v, v);
    $|  }
    $|  part[lid.x] = ss;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var sum = 0.0;
    $|    for (var t = 0u; t < 256u; t = t + 1u) { sum = sum + part[t]; }
    $|    inv_s = 1.0 / sqrt(sum / f32(N) + EPS);
    $|  }
    $|  workgroupBarrier();
    $|  let inv = inv_s;
    $|  for (var t = lid.x; t < N4; t = t + 256u) {
    $|    y[base + t] = x[base + t] * inv * w[t];
    $|  }
    $|}
  src
}

///|
/// NeoX RoPE over x [rows, heads*2*half], in place: for pair j of head h
/// at position pos0 + row, (a, b) = (x[.. j], x[.. j+half]) rotates by
/// (cos[pos, j], sin[pos, j]) from precomputed tables [max_pos, half].
/// One thread per pair. Grid: (ceil(rows*heads*half / 256), 1).
pub fn rope_shader(rows : Int, heads : Int, half : Int) -> String {
  let src =
    $|@group(0) @binding(0) var<storage, read_write> x : array<f32>;
    $|@group(0) @binding(1) var<storage, read> cost : array<f32>;
    $|@group(0) @binding(2) var<storage, read> sint : array<f32>;
    $|@group(0) @binding(3) var<storage, read> params : array<u32>;
    $|
    $|const S : u32 = \{rows}u;
    $|const H : u32 = \{heads}u;
    $|const HALF : u32 = \{half}u;
    $|const HD : u32 = \{half * 2}u;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $|  let i = gid.x;
    $|  if (i >= S * H * HALF) { return; }
    $|  let row = i / (H * HALF);
    $|  let r = i % (H * HALF);
    $|  let head = r / HALF;
    $|  let j = r % HALF;
    $|  let pos = params[0] + row;
    $|  let base = (row * H + head) * HD + j;
    $|  let c = cost[pos * HALF + j];
    $|  let s = sint[pos * HALF + j];
    $|  let a = x[base];
    $|  let b = x[base + HALF];
    $|  x[base] = a * c - b * s;
    $|  x[base + HALF] = a * s + b * c;
    $|}
  src
}

///|
/// y[i] = silu(g[i]) * u[i], vec4 elementwise. Grid: (ceil(n/1024), 1).
/// In-place safe on any argument.
pub fn silu_mul_shader(n : Int) -> String {
  let n4 = n / 4
  let src =
    $|// read_write so in-place calls (y == g) bind one storage usage.
    $|@group(0) @binding(0) var<storage, read_write> g : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> u : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read_write> y : array<vec4<f32>>;
    $|
    $|const N4 : u32 = \{n4}u;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $|  let i = gid.x;
    $|  if (i >= N4) { return; }
    $|  let gv = g[i];
    $|  y[i] = gv / (vec4<f32>(1.0) + exp(-gv)) * u[i];
    $|}
  src
}

///|
/// x[i] += y[i], vec4 elementwise. Grid: (ceil(n/1024), 1).
pub fn add_shader(n : Int) -> String {
  let n4 = n / 4
  let src =
    $|@group(0) @binding(0) var<storage, read_write> x : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> y : array<vec4<f32>>;
    $|
    $|const N4 : u32 = \{n4}u;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $|  let i = gid.x;
    $|  if (i >= N4) { return; }
    $|  x[i] = x[i] + y[i];
    $|}
  src
}

///|
/// Multi-row causal GQA attention, one workgroup per (q head, query row):
/// q [rows, hq*hd], kv caches [max_pos, hkv*hd], out [rows, hq*hd].
/// Row r attends positions 0..pos0+r (params[0] = pos0) — one kernel for
/// both prefill (rows = prompt, pos0 = 0) and decode (rows = 1).
///
/// 128 threads: stage q row, score all m positions (thread-strided rows
/// of the kv head), shared max/sum reductions, then thread-per-dim
/// weighted V accumulation. Scores live in shared memory (max_pos * 4B).
/// Grid: (hq, rows). Requires hd == 128 (thread-per-dim mapping).
pub fn attn_rows_shader(
  hq : Int,
  hkv : Int,
  hd : Int,
  max_pos : Int,
  scale : Double,
) -> String {
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> kc : array<f32>;
    $|@group(0) @binding(2) var<storage, read> vc : array<f32>;
    $|@group(0) @binding(3) var<storage, read_write> outp : array<f32>;
    $|@group(0) @binding(4) var<storage, read> params : array<u32>;
    $|
    $|const HQ : u32 = \{hq}u;
    $|const HKV : u32 = \{hkv}u;
    $|const HD : u32 = \{hd}u;
    $|const GROUP : u32 = \{hq / hkv}u;
    $|const SCALE : f32 = \{scale};
    $|
    $|var<workgroup> qs : array<f32, \{hd}>;
    $|var<workgroup> sc : array<f32, \{max_pos}>;
    $|var<workgroup> part : array<f32, 128>;
    $|var<workgroup> bcast : array<f32, 2>; // 0: max, 1: sum
    $|
    $|@compute @workgroup_size(128)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let head = wid.x;
    $|  let row = wid.y;
    $|  let kvh = head / GROUP;
    $|  let m = params[0] + row + 1u;
    $|  qs[lid.x] = q[(row * HQ + head) * HD + lid.x];
    $|  workgroupBarrier();
    $|  // scores
    $|  var lmax = -3.402823466e+38;
    $|  for (var p = lid.x; p < m; p = p + 128u) {
    $|    let kb = (p * HKV + kvh) * HD;
    $|    var acc = 0.0;
    $|    for (var d = 0u; d < HD; d = d + 1u) {
    $|      acc = acc + qs[d] * kc[kb + d];
    $|    }
    $|    let s = acc * SCALE;
    $|    sc[p] = s;
    $|    lmax = max(lmax, s);
    $|  }
    $|  part[lid.x] = lmax;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var mx = -3.402823466e+38;
    $|    for (var t = 0u; t < 128u; t = t + 1u) { mx = max(mx, part[t]); }
    $|    bcast[0] = mx;
    $|  }
    $|  workgroupBarrier();
    $|  let mx = bcast[0];
    $|  var lsum = 0.0;
    $|  for (var p = lid.x; p < m; p = p + 128u) {
    $|    let e = exp(sc[p] - mx);
    $|    sc[p] = e;
    $|    lsum = lsum + e;
    $|  }
    $|  part[lid.x] = lsum;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var sm = 0.0;
    $|    for (var t = 0u; t < 128u; t = t + 1u) { sm = sm + part[t]; }
    $|    bcast[1] = sm;
    $|  }
    $|  workgroupBarrier();
    $|  let inv = 1.0 / bcast[1];
    $|  // weighted V sum, thread per output dim
    $|  var acc = 0.0;
    $|  for (var p = 0u; p < m; p = p + 1u) {
    $|    acc = acc + sc[p] * vc[(p * HKV + kvh) * HD + lid.x];
    $|  }
    $|  outp[(row * HQ + head) * HD + lid.x] = acc * inv;
    $|}
  src
}
