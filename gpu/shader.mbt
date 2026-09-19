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
/// GEMV kernel, bf16 weights: y[out] = W[out, in] @ x[in], W row-major
/// bf16 pairs packed as u32 (exact upcast in-kernel: u<<16 / u&0xffff0000).
///
/// 256-thread workgroup covers 8 output rows (one 32-lane warp per row):
/// x staged in shared as f32 (<= 12KB for in <= 3072), lanes stream W rows
/// as coalesced u32 reads, then a per-warp shared reduction.
/// Grid: (ceil(out/8), 1). Requires in % 2 == 0.
pub fn gemv_shader(out : Int, inn : Int) -> String {
  let in8 = inn / 8
  let src =
    $|@group(0) @binding(0) var<storage, read> w : array<vec4<u32>>;
    $|@group(0) @binding(1) var<storage, read> x : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read_write> y : array<f32>;
    $|
    $|const OUT : u32 = \{out}u;
    $|const IN8 : u32 = \{in8}u;
    $|
    $|var<workgroup> part : array<f32, 512>;
    $|
    $|@compute @workgroup_size(512)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let rg = lid.x / 32u; // row group (warp) 0..15
    $|  let lane = lid.x % 32u;
    $|  let o = wid.x * 16u + rg;
    $|  let oo = min(o, OUT - 1u);
    $|  var acc = 0.0;
    $|  // x is 4KB and read by every workgroup: L1-resident; skip the
    $|  // shared staging + barrier and read it straight from global.
    $|  for (var t = lane; t < IN8; t = t + 32u) {
    $|    let u4 = w[oo * IN8 + t]; // 8 bf16
    $|    let x0 = x[t * 2u];
    $|    let x1 = x[t * 2u + 1u];
    $|    acc = acc + bitcast<f32>(u4.x << 16u) * x0.x
    $|      + bitcast<f32>(u4.x & 0xffff0000u) * x0.y
    $|      + bitcast<f32>(u4.y << 16u) * x0.z
    $|      + bitcast<f32>(u4.y & 0xffff0000u) * x0.w
    $|      + bitcast<f32>(u4.z << 16u) * x1.x
    $|      + bitcast<f32>(u4.z & 0xffff0000u) * x1.y
    $|      + bitcast<f32>(u4.w << 16u) * x1.z
    $|      + bitcast<f32>(u4.w & 0xffff0000u) * x1.w;
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
/// GEMM kernel, bf16 weights: Y[s, out] = X[s, in] @ W[out, in]^T.
/// X row-major f32, W row-major bf16 (u32 pairs), Y f32.
///
/// 16x16 output tile per 256-thread workgroup; k streamed in 8-element
/// chunks: X tile staged as f32 (2KB), W tile as raw u32 (1KB), unpacked
/// at use. Edges zero-padded via select. Requires in % 8 == 0.
pub fn matmul_shader(s : Int, out : Int, inn : Int) -> String {
  let src =
    $|@group(0) @binding(0) var<storage, read> x : array<f32>;
    $|@group(0) @binding(1) var<storage, read> w : array<u32>;
    $|@group(0) @binding(2) var<storage, read_write> y : array<f32>;
    $|
    $|const S : u32 = \{s}u;
    $|const OUT : u32 = \{out}u;
    $|const IN : u32 = \{inn}u;
    $|const IN2 : u32 = \{inn / 2}u;
    $|
    $|var<workgroup> xs : array<f32, 128>; // 16 rows x 8 elems
    $|var<workgroup> ws : array<u32, 64>; // 16 rows x 4 u32 (8 bf16)
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
    $|  for (var k0 = 0u; k0 < IN; k0 = k0 + 8u) {
    $|    for (var t = lid.x; t < 128u; t = t + 256u) {
    $|      let r = t / 8u;
    $|      let c = t % 8u;
    $|      let gi = row0 + r;
    $|      xs[t] = select(0.0, x[gi * IN + k0 + c], gi < S);
    $|    }
    $|    for (var t = lid.x; t < 64u; t = t + 256u) {
    $|      let r = t / 4u;
    $|      let c = t % 4u;
    $|      let gj = col0 + r;
    $|      ws[t] = select(0u, w[gj * IN2 + k0 / 2u + c], gj < OUT);
    $|    }
    $|    workgroupBarrier();
    $|    for (var kk = 0u; kk < 4u; kk = kk + 1u) {
    $|      let u = ws[tx * 4u + kk];
    $|      acc = acc + xs[ty * 8u + kk * 2u] * bitcast<f32>(u << 16u)
    $|        + xs[ty * 8u + kk * 2u + 1u] * bitcast<f32>(u & 0xffff0000u);
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
/// Flash prefill attention (tiled online softmax), causal GQA over KV
/// caches [max_pos, hkv*hd]: q [rows, hq*hd] -> out [rows, hq*hd].
/// Row i attends keys 0..params[0]+i. One workgroup per (64-row block,
/// head); thread-per-row with the q row and output accumulator textually
/// unrolled into registers. K/V tiles (BC=8 rows) are staged cooperatively
/// in shared memory: 2*8*hd*4 = 8KB at hd=128, under the 16KB workgroup
/// limit. The score matrix is never materialized (neither global nor
/// shared) — no max_pos-sized scratch, so the kernel imposes no context cap.
/// Grid: (ceil(rows/64), hq). Requires hd <= 128.
pub fn attn_prefill_shader(
  rows : Int,
  hq : Int,
  hkv : Int,
  hd : Int,
  scale : Double,
) -> String {
  let qload = unroll_dv(hd, fn(buf, t) {
    buf.write_string("qr[\{t}] = q[(i * HQ + head) * HD + \{t}u];")
  })
  let dot = dot4(hd, "qr", "ks[j * HD + ")
  let accum = unroll_dv(hd, fn(buf, c) {
    buf.write_string("o[\{c}] = o[\{c}] * alpha + p * vs[j * HD + \{c}u];")
  })
  let store = unroll_dv(hd, fn(buf, c) {
    buf.write_string("outp[(i * HQ + head) * HD + \{c}u] = o[\{c}] * inv;")
  })
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> kc : array<f32>;
    $|@group(0) @binding(2) var<storage, read> vc : array<f32>;
    $|@group(0) @binding(3) var<storage, read_write> outp : array<f32>;
    $|@group(0) @binding(4) var<storage, read> params : array<u32>;
    $|
    $|const ROWS : u32 = \{rows}u;
    $|const HQ : u32 = \{hq}u;
    $|const HKV : u32 = \{hkv}u;
    $|const HD : u32 = \{hd}u;
    $|const GROUP : u32 = \{hq / hkv}u;
    $|const SCALE : f32 = \{scale};
    $|const BC : u32 = 8u;
    $|const WG : u32 = 64u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|var<workgroup> ks : array<f32, BC * HD>;
    $|var<workgroup> vs : array<f32, BC * HD>;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let head = wid.y;
    $|  let kvh = head / GROUP;
    $|  let i = wid.x * WG + lid.x;
    $|  let base = params[0];
    $|  // deepest causal key bound among this block's rows
    $|  let j_end = base + min(wid.x * WG + WG, ROWS);
    $|  var o : array<f32, HD>;
    $|  var qr : array<f32, HD>;
    $|  var m_run = NEG_BIG;
    $|  var l_run = 0.0;
    $|  if (i < ROWS) {
    $|    \{qload}
    $|  }
    $|  for (var j0 = 0u; j0 < j_end; j0 = j0 + BC) {
    $|    for (var t = lid.x; t < BC * HD; t = t + WG) {
    $|      let gj = j0 + t / HD;
    $|      ks[t] = select(0.0, kc[(gj * HKV + kvh) * HD + t % HD], gj < j_end);
    $|    }
    $|    for (var t = lid.x; t < BC * HD; t = t + WG) {
    $|      let gj = j0 + t / HD;
    $|      vs[t] = select(0.0, vc[(gj * HKV + kvh) * HD + t % HD], gj < j_end);
    $|    }
    $|    workgroupBarrier();
    $|    if (i < ROWS) {
    $|      let mlim = base + i + 1u;
    $|      for (var j = 0u; j < BC; j = j + 1u) {
    $|        if (j0 + j >= mlim) { break; }
    $|        \{dot}
    $|        let s = s_dot * SCALE;
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
    $|  if (i < ROWS) {
    $|    let inv = 1.0 / l_run;
    $|    \{store}
    $|  }
    $|}
  src
}

///|
/// Flash-decoding split-KV attention, kernel 1 (partials), decode shape:
/// q [1, hq*hd] over KV caches [max_pos, hkv*hd], m = params[0] + 1 keys.
/// Grid: (splits, hq). 128 threads stage the q head in shared, score their
/// key chunk (shared max/exp/sum passes over a max_pos/splits-entry array
/// — no global score buffer, no max_pos-sized scratch), then accumulate
/// the UNNORMALIZED weighted V (thread-per-dim) and write a partial
/// record (m, l, acc[hd]) at part[(head*S + sp) * (hd+4)]. Empty splits
/// (j0 >= m) write m = NEG_BIG, l = 0; the combine kernel skips them via
/// exp(m - M) = 0.
pub fn attn_decode_shader(
  hq : Int,
  hkv : Int,
  hd : Int,
  max_pos : Int,
  splits : Int,
  scale : Double,
) -> String {
  let maxchunk = (max_pos + splits - 1) / splits
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> kc : array<f32>;
    $|@group(0) @binding(2) var<storage, read> vc : array<f32>;
    $|@group(0) @binding(3) var<storage, read_write> part : array<f32>;
    $|@group(0) @binding(4) var<storage, read> params : array<u32>;
    $|
    $|const HQ : u32 = \{hq}u;
    $|const HKV : u32 = \{hkv}u;
    $|const HD : u32 = \{hd}u;
    $|const GROUP : u32 = \{hq / hkv}u;
    $|const SCALE : f32 = \{scale};
    $|const S : u32 = \{splits}u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|var<workgroup> qs : array<f32, HD>;
    $|var<workgroup> sc : array<f32, \{maxchunk}>;
    $|var<workgroup> red : array<f32, 128>;
    $|var<workgroup> bcast : array<f32, 2>; // 0: max, 1: sum
    $|
    $|@compute @workgroup_size(128)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let sp = wid.x;
    $|  let head = wid.y;
    $|  let kvh = head / GROUP;
    $|  let m = params[0] + 1u;
    $|  let chunk = (m + S - 1u) / S;
    $|  let j0 = sp * chunk;
    $|  let j1 = min(j0 + chunk, m);
    $|  qs[lid.x] = q[head * HD + lid.x];
    $|  workgroupBarrier();
    $|  var lmax = NEG_BIG;
    $|  for (var p = j0 + lid.x; p < j1; p = p + 128u) {
    $|    let kb = (p * HKV + kvh) * HD;
    $|    var acc = 0.0;
    $|    for (var d = 0u; d < HD; d = d + 1u) {
    $|      acc = acc + qs[d] * kc[kb + d];
    $|    }
    $|    let s = acc * SCALE;
    $|    sc[p - j0] = s;
    $|    lmax = max(lmax, s);
    $|  }
    $|  red[lid.x] = lmax;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var mx = NEG_BIG;
    $|    for (var t = 0u; t < 128u; t = t + 1u) { mx = max(mx, red[t]); }
    $|    bcast[0] = mx;
    $|  }
    $|  workgroupBarrier();
    $|  let mx = bcast[0];
    $|  var lsum = 0.0;
    $|  for (var p = j0 + lid.x; p < j1; p = p + 128u) {
    $|    let e = exp(sc[p - j0] - mx);
    $|    sc[p - j0] = e;
    $|    lsum = lsum + e;
    $|  }
    $|  red[lid.x] = lsum;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var sm = 0.0;
    $|    for (var t = 0u; t < 128u; t = t + 1u) { sm = sm + red[t]; }
    $|    bcast[1] = sm;
    $|  }
    $|  workgroupBarrier();
    $|  // unnormalized weighted V, thread per output dim (sc reads are
    $|  // covered by the barrier above)
    $|  var acc = 0.0;
    $|  for (var p = j0; p < j1; p = p + 1u) {
    $|    acc = acc + sc[p - j0] * vc[(p * HKV + kvh) * HD + lid.x];
    $|  }
    $|  let rec = (head * S + sp) * (HD + 4u);
    $|  if (lid.x == 0u) {
    $|    part[rec] = mx;
    $|    part[rec + 1u] = bcast[1];
    $|  }
    $|  part[rec + 2u + lid.x] = acc;
    $|}
  src
}

///|
/// Flash-decoding kernel 2: merge the S per-split partials into the final
/// output. One workgroup of hd threads per head (thread-per-dim): the
/// cross-split online-softmax rescale w_s = exp(m_s - M), out = sum(w_s *
/// acc_s) / sum(w_s * l_s). Grid: (hq, 1).
pub fn attn_combine_shader(hd : Int, splits : Int) -> String {
  let src =
    $|@group(0) @binding(0) var<storage, read> part : array<f32>;
    $|@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
    $|
    $|const HD : u32 = \{hd}u;
    $|const S : u32 = \{splits}u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|@compute @workgroup_size(\{hd})
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let head = wid.x;
    $|  let d = lid.x;
    $|  var mmax = NEG_BIG;
    $|  for (var s = 0u; s < S; s = s + 1u) {
    $|    mmax = max(mmax, part[(head * S + s) * (HD + 4u)]);
    $|  }
    $|  var ltot = 0.0;
    $|  var acc = 0.0;
    $|  for (var s = 0u; s < S; s = s + 1u) {
    $|    let rec = (head * S + s) * (HD + 4u);
    $|    let w = exp(part[rec] - mmax);
    $|    ltot = ltot + w * part[rec + 1u];
    $|    acc = acc + w * part[rec + 2u + d];
    $|  }
    $|  outp[head * HD + d] = acc / ltot;
    $|}
  src
}

///|
/// Embedding gather with bf16 unpack: for each row, dst[row, i] =
/// f32(emb[ids[row] * n + i]). One thread per u32 (2 elements).
/// Grid: (ceil(rows * n/2 / 256), 1). Requires n % 2 == 0.
pub fn embed_rows_shader(rows : Int, n : Int) -> String {
  let n2 = n / 2
  let src =
    $|@group(0) @binding(0) var<storage, read> emb : array<u32>;
    $|@group(0) @binding(1) var<storage, read> ids : array<u32>;
    $|@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
    $|
    $|const R : u32 = \{rows}u;
    $|const N : u32 = \{n}u;
    $|const N2 : u32 = \{n2}u;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $|  let i = gid.x;
    $|  if (i >= R * N2) { return; }
    $|  let row = i / N2;
    $|  let c = i % N2;
    $|  let u = emb[ids[row] * N2 + c];
    $|  dst[row * N + c * 2u] = bitcast<f32>(u << 16u);
    $|  dst[row * N + c * 2u + 1u] = bitcast<f32>(u & 0xffff0000u);
    $|}
  src
}

///|
/// Fused QKV GEMV (decode): rows 0..nq-1 -> qout, nq..nq+nkv-1 -> kout,
/// nq+nkv.. -> vout, where vout is the KV cache at params[0] (v is cached
/// before use, so the projection writes it directly). bf16 weights.
/// Same warp-per-row structure as gemv_shader. Grid: (ceil((nq+2nkv)/8), 1).
pub fn qkv_fused_shader(nq : Int, nkv : Int, inn : Int) -> String {
  let in2 = inn / 2
  let src =
    $|@group(0) @binding(0) var<storage, read> x : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> wq : array<u32>;
    $|@group(0) @binding(2) var<storage, read> wk : array<u32>;
    $|@group(0) @binding(3) var<storage, read> wv : array<u32>;
    $|@group(0) @binding(4) var<storage, read_write> qout : array<f32>;
    $|@group(0) @binding(5) var<storage, read_write> kout : array<f32>;
    $|@group(0) @binding(6) var<storage, read_write> vc : array<f32>;
    $|@group(0) @binding(7) var<storage, read> params : array<u32>;
    $|
    $|const NQ : u32 = \{nq}u;
    $|const NKV : u32 = \{nkv}u;
    $|const IN2 : u32 = \{in2}u;
    $|
    $|var<workgroup> xs : array<f32, 3072>;
    $|var<workgroup> part : array<f32, 256>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let rg = lid.x / 32u;
    $|  let lane = lid.x % 32u;
    $|  let o = wid.x * 8u + rg;
    $|  let oo = min(o, NQ + 2u * NKV - 1u);
    $|  for (var t = lid.x; t < IN2 * 2u; t = t + 256u) {
    $|    xs[t] = x[t / 4u][t % 4u];
    $|  }
    $|  workgroupBarrier();
    $|  var acc = 0.0;
    $|  if (oo < NQ) {
    $|    for (var t = lane; t < IN2; t = t + 32u) {
    $|      let u = wq[oo * IN2 + t];
    $|      acc = acc + bitcast<f32>(u << 16u) * xs[t * 2u]
    $|        + bitcast<f32>(u & 0xffff0000u) * xs[t * 2u + 1u];
    $|    }
    $|  } else if (oo < NQ + NKV) {
    $|    let r = oo - NQ;
    $|    for (var t = lane; t < IN2; t = t + 32u) {
    $|      let u = wk[r * IN2 + t];
    $|      acc = acc + bitcast<f32>(u << 16u) * xs[t * 2u]
    $|        + bitcast<f32>(u & 0xffff0000u) * xs[t * 2u + 1u];
    $|    }
    $|  } else {
    $|    let r = oo - NQ - NKV;
    $|    for (var t = lane; t < IN2; t = t + 32u) {
    $|      let u = wv[r * IN2 + t];
    $|      acc = acc + bitcast<f32>(u << 16u) * xs[t * 2u]
    $|        + bitcast<f32>(u & 0xffff0000u) * xs[t * 2u + 1u];
    $|    }
    $|  }
    $|  part[lid.x] = acc;
    $|  workgroupBarrier();
    $|  if (lane == 0u && o < NQ + 2u * NKV) {
    $|    var sum = 0.0;
    $|    for (var t = 0u; t < 32u; t = t + 1u) {
    $|      sum = sum + part[rg * 32u + t];
    $|    }
    $|    if (o < NQ) {
    $|      qout[o] = sum;
    $|    } else if (o < NQ + NKV) {
    $|      kout[o - NQ] = sum;
    $|    } else {
    $|      vc[params[0] * NKV + o - NQ - NKV] = sum;
    $|    }
    $|  }
    $|}
  src
}

///|
/// Fused gate+up GEMV (decode): rows 0..n-1 -> gout from wg, n..2n-1 ->
/// uout from wu. bf16 weights. Grid: (ceil(2n/8), 1).
pub fn gemv2_fused_shader(n : Int, inn : Int) -> String {
  let in2 = inn / 2
  let src =
    $|@group(0) @binding(0) var<storage, read> x : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> wg : array<u32>;
    $|@group(0) @binding(2) var<storage, read> wu : array<u32>;
    $|@group(0) @binding(3) var<storage, read_write> gout : array<f32>;
    $|@group(0) @binding(4) var<storage, read_write> uout : array<f32>;
    $|
    $|const N : u32 = \{n}u;
    $|const IN2 : u32 = \{in2}u;
    $|
    $|var<workgroup> xs : array<f32, 3072>;
    $|var<workgroup> part : array<f32, 256>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let rg = lid.x / 32u;
    $|  let lane = lid.x % 32u;
    $|  let o = wid.x * 8u + rg;
    $|  let oo = min(o, 2u * N - 1u);
    $|  for (var t = lid.x; t < IN2 * 2u; t = t + 256u) {
    $|    xs[t] = x[t / 4u][t % 4u];
    $|  }
    $|  workgroupBarrier();
    $|  var acc = 0.0;
    $|  if (oo < N) {
    $|    for (var t = lane; t < IN2; t = t + 32u) {
    $|      let u = wg[oo * IN2 + t];
    $|      acc = acc + bitcast<f32>(u << 16u) * xs[t * 2u]
    $|        + bitcast<f32>(u & 0xffff0000u) * xs[t * 2u + 1u];
    $|    }
    $|  } else {
    $|    let r = oo - N;
    $|    for (var t = lane; t < IN2; t = t + 32u) {
    $|      let u = wu[r * IN2 + t];
    $|      acc = acc + bitcast<f32>(u << 16u) * xs[t * 2u]
    $|        + bitcast<f32>(u & 0xffff0000u) * xs[t * 2u + 1u];
    $|    }
    $|  }
    $|  part[lid.x] = acc;
    $|  workgroupBarrier();
    $|  if (lane == 0u && o < 2u * N) {
    $|    var sum = 0.0;
    $|    for (var t = 0u; t < 32u; t = t + 1u) {
    $|      sum = sum + part[rg * 32u + t];
    $|    }
    $|    if (o < N) {
    $|      gout[o] = sum;
    $|    } else {
    $|      uout[o - N] = sum;
    $|    }
    $|  }
    $|}
  src
}

///|
/// Fused per-head RMSNorm + NeoX RoPE for q and k (decode, one row).
/// Workgroup h < HQ handles q head h in place; h >= HQ handles k head
/// h-HQ, writing the normed+rotated result directly into the KV cache at
/// params[0]. 64 threads = 64 pairs; each thread norms AND rotates its
/// pair (both halves loaded up front). bf16... no — norm weights are f32.
/// Grid: (HQ + HKV, 1). Requires hd = 128.
pub fn qknorm_rope_shader(hq : Int, hkv : Int, hd : Int, eps : Float) -> String {
  let half = hd / 2
  let src =
    $|@group(0) @binding(0) var<storage, read_write> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> k : array<f32>;
    $|@group(0) @binding(2) var<storage, read> qnw : array<f32>;
    $|@group(0) @binding(3) var<storage, read> knw : array<f32>;
    $|@group(0) @binding(4) var<storage, read_write> kc : array<f32>;
    $|@group(0) @binding(5) var<storage, read> cost : array<f32>;
    $|@group(0) @binding(6) var<storage, read> sint : array<f32>;
    $|@group(0) @binding(7) var<storage, read> params : array<u32>;
    $|
    $|const HQ : u32 = \{hq}u;
    $|const HD : u32 = \{hd}u;
    $|const HALF : u32 = \{half}u;
    $|const EPS : f32 = \{eps};
    $|
    $|var<workgroup> part : array<f32, 64>;
    $|var<workgroup> inv_s : f32;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let is_q = wid.x < HQ;
    $|  let head = select(wid.x - HQ, wid.x, is_q);
    $|  let base = head * HD + lid.x;
    $|  let v1 = select(k[base], q[base], is_q);
    $|  let v2 = select(k[base + HALF], q[base + HALF], is_q);
    $|  part[lid.x] = v1 * v1 + v2 * v2;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var sum = 0.0;
    $|    for (var t = 0u; t < 64u; t = t + 1u) { sum = sum + part[t]; }
    $|    inv_s = 1.0 / sqrt(sum / f32(HD) + EPS);
    $|  }
    $|  workgroupBarrier();
    $|  let inv = inv_s;
    $|  let n1 = v1 * inv * select(knw[lid.x], qnw[lid.x], is_q);
    $|  let n2 = v2 * inv * select(knw[lid.x + HALF], qnw[lid.x + HALF], is_q);
    $|  let pos = params[0];
    $|  let c = cost[pos * HALF + lid.x];
    $|  let s = sint[pos * HALF + lid.x];
    $|  let o1 = n1 * c - n2 * s;
    $|  let o2 = n1 * s + n2 * c;
    $|  if (is_q) {
    $|    q[base] = o1;
    $|    q[base + HALF] = o2;
    $|  } else {
    $|    let cb = (pos * \{hkv}u + head) * HD + lid.x;
    $|    kc[cb] = o1;
    $|    kc[cb + HALF] = o2;
    $|  }
    $|}
  src
}

///|
/// Fused residual add + RMSNorm: x[r] += d[r]; y[r] = rmsnorm(x[r]) * w.
/// One workgroup per row, 256 threads vec4. Grid: (rows, 1).
pub fn add_rmsnorm_shader(n : Int, eps : Float) -> String {
  let n4 = n / 4
  let src =
    $|@group(0) @binding(0) var<storage, read_write> x : array<vec4<f32>>;
    $|@group(0) @binding(1) var<storage, read> d : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read> w : array<vec4<f32>>;
    $|@group(0) @binding(3) var<storage, read_write> y : array<vec4<f32>>;
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
    $|    let v = x[base + t] + d[base + t];
    $|    x[base + t] = v;
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
/// Down-projection GEMV with fused activation: x staging computes
/// xs[i] = silu(g[i]) * u[i] on the fly. bf16 weights, warp-per-row.
/// Grid: (ceil(out/8), 1). Requires inn % 2 == 0.
pub fn gemv_silu_shader(out : Int, inn : Int) -> String {
  let in2 = inn / 2
  let src =
    $|@group(0) @binding(0) var<storage, read> w : array<u32>;
    $|@group(0) @binding(1) var<storage, read> g : array<vec4<f32>>;
    $|@group(0) @binding(2) var<storage, read> u : array<vec4<f32>>;
    $|@group(0) @binding(3) var<storage, read_write> y : array<f32>;
    $|
    $|const OUT : u32 = \{out}u;
    $|const IN2 : u32 = \{in2}u;
    $|
    $|var<workgroup> xs : array<f32, 3072>;
    $|var<workgroup> part : array<f32, 256>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let rg = lid.x / 32u;
    $|  let lane = lid.x % 32u;
    $|  let o = wid.x * 8u + rg;
    $|  let oo = min(o, OUT - 1u);
    $|  for (var t = lid.x; t < IN2 * 2u; t = t + 256u) {
    $|    let gv = g[t / 4u][t % 4u];
    $|    xs[t] = gv / (1.0 + exp(-gv)) * u[t / 4u][t % 4u];
    $|  }
    $|  workgroupBarrier();
    $|  var acc = 0.0;
    $|  for (var t = lane; t < IN2; t = t + 32u) {
    $|    let uu = w[oo * IN2 + t];
    $|    acc = acc + bitcast<f32>(uu << 16u) * xs[t * 2u]
    $|      + bitcast<f32>(uu & 0xffff0000u) * xs[t * 2u + 1u];
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
/// Argmax phase 1: each 256-thread workgroup scans CHUNK consecutive
/// logits; thread 0 serial-reduces the 256 (val, idx) partials.
/// part_v[wid] / part_i[wid] hold the workgroup winner. Grid: (wgs, 1).
pub fn argmax1_shader(n : Int, chunk : Int) -> String {
  let src =
    $|@group(0) @binding(0) var<storage, read> x : array<f32>;
    $|@group(0) @binding(1) var<storage, read_write> part_v : array<f32>;
    $|@group(0) @binding(2) var<storage, read_write> part_i : array<f32>;
    $|
    $|const N : u32 = \{n}u;
    $|const CHUNK : u32 = \{chunk}u;
    $|
    $|var<workgroup> pv : array<f32, 256>;
    $|var<workgroup> pi : array<f32, 256>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  var bv = -3.402823466e+38;
    $|  var bi = 0u;
    $|  let base = wid.x * CHUNK;
    $|  for (var t = lid.x; t < CHUNK; t = t + 256u) {
    $|    let i = base + t;
    $|    if (i < N) {
    $|      let v = x[i];
    $|      if (v > bv) { bv = v; bi = i; }
    $|    }
    $|  }
    $|  pv[lid.x] = bv;
    $|  pi[lid.x] = f32(bi);
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var mv = -3.402823466e+38;
    $|    var mi = 0.0;
    $|    for (var t = 0u; t < 256u; t = t + 1u) {
    $|      if (pv[t] > mv) { mv = pv[t]; mi = pi[t]; }
    $|    }
    $|    part_v[wid.x] = mv;
    $|    part_i[wid.x] = mi;
    $|  }
    $|}
  src
}

///|
/// Argmax phase 2: one 256-thread workgroup reduces WGS partials; out[0]
/// gets the winning index as f32 (vocab < 2^24, exact).
pub fn argmax2_shader(wgs : Int) -> String {
  let src =
    $|@group(0) @binding(0) var<storage, read> part_v : array<f32>;
    $|@group(0) @binding(1) var<storage, read> part_i : array<f32>;
    $|@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
    $|
    $|const WGS : u32 = \{wgs}u;
    $|
    $|var<workgroup> pv : array<f32, 256>;
    $|var<workgroup> pi : array<f32, 256>;
    $|
    $|@compute @workgroup_size(256)
    $|fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
    $|  var bv = -3.402823466e+38;
    $|  var bi = 0.0;
    $|  for (var t = lid.x; t < WGS; t = t + 256u) {
    $|    let v = part_v[t];
    $|    if (v > bv) { bv = v; bi = part_i[t]; }
    $|  }
    $|  pv[lid.x] = bv;
    $|  pi[lid.x] = bi;
    $|  workgroupBarrier();
    $|  if (lid.x == 0u) {
    $|    var mv = -3.402823466e+38;
    $|    var mi = 0.0;
    $|    for (var t = 0u; t < 256u; t = t + 1u) {
    $|      if (pv[t] > mv) { mv = pv[t]; mi = pi[t]; }
    $|    }
    $|    outp[0] = mi;
    $|  }
    $|}
  src
}

///|
/// Library 4D flash attention (tiled online softmax): q [B,H,Sq,D],
/// k [B,HKV,Skv,D], v [B,HKV,Skv,DV] -> out [B,H,Sq,DV], optional
/// bottom-right causal mask (CAUSAL=1: row i sees keys 0..=i+Skv-Sq).
/// One workgroup per (64-row block, batch*head); thread-per-row with the
/// q row and output accumulator textually unrolled into registers. K/V
/// tiles (BC=8 rows) staged cooperatively in shared memory. Scores are
/// never materialized. Grid: (ceil(Sq/64), B*H). Requires D, DV <= 128.
pub fn attn_4d_shader(
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  dv : Int,
  causal : Bool,
  scale : Double,
) -> String {
  let qload = unroll_dv(d, fn(buf, t) {
    buf.write_string("qr[\{t}] = q[qoff + i * D + \{t}u];")
  })
  let dot = dot4(d, "qr", "ks[j * D + ")
  let accum = unroll_dv(dv, fn(buf, c) {
    buf.write_string("o[\{c}] = o[\{c}] * alpha + p * vs[j * DV + \{c}u];")
  })
  let store = unroll_dv(dv, fn(buf, c) {
    buf.write_string("outp[(bh * SQ + i) * DV + \{c}u] = o[\{c}] * inv;")
  })
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> kc : array<f32>;
    $|@group(0) @binding(2) var<storage, read> vc : array<f32>;
    $|@group(0) @binding(3) var<storage, read_write> outp : array<f32>;
    $|
    $|const B : u32 = \{b}u;
    $|const H : u32 = \{h}u;
    $|const HKV : u32 = \{hkv}u;
    $|const GROUP : u32 = \{h / hkv}u;
    $|const SQ : u32 = \{sq}u;
    $|const SKV : u32 = \{skv}u;
    $|const D : u32 = \{d}u;
    $|const DV : u32 = \{dv}u;
    $|const DELTA : u32 = \{skv - sq}u; // SKV >= SQ when causal
    $|const CAUSAL : bool = \{causal};
    $|const SCALE : f32 = \{scale};
    $|const BC : u32 = 8u;
    $|const WG : u32 = 64u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|var<workgroup> ks : array<f32, BC * D>;
    $|var<workgroup> vs : array<f32, BC * DV>;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let bh = wid.y;
    $|  let bat = bh / H;
    $|  let head = bh % H;
    $|  let kvh = head / GROUP;
    $|  let i = wid.x * WG + lid.x;
    $|  let qoff = bh * SQ * D;
    $|  let koff = (bat * HKV + kvh) * SKV * D;
    $|  let voff = (bat * HKV + kvh) * SKV * DV;
    $|  // deepest key bound among this block's rows
    $|  var j_end = SKV;
    $|  if (CAUSAL) {
    $|    j_end = min(min(wid.x * WG + WG, SQ) + DELTA, SKV);
    $|  }
    $|  var o : array<f32, DV>;
    $|  var qr : array<f32, D>;
    $|  var m_run = NEG_BIG;
    $|  var l_run = 0.0;
    $|  if (i < SQ) {
    $|    \{qload}
    $|  }
    $|  for (var j0 = 0u; j0 < j_end; j0 = j0 + BC) {
    $|    for (var t = lid.x; t < BC * D; t = t + WG) {
    $|      let gj = j0 + t / D;
    $|      ks[t] = select(0.0, kc[koff + gj * D + t % D], gj < j_end);
    $|    }
    $|    for (var t = lid.x; t < BC * DV; t = t + WG) {
    $|      let gj = j0 + t / DV;
    $|      vs[t] = select(0.0, vc[voff + gj * DV + t % DV], gj < j_end);
    $|    }
    $|    workgroupBarrier();
    $|    if (i < SQ) {
    $|      var mlim = SKV;
    $|      if (CAUSAL) {
    $|        mlim = min(i + DELTA + 1u, SKV);
    $|      }
    $|      for (var j = 0u; j < BC; j = j + 1u) {
    $|        if (j0 + j >= mlim) { break; }
    $|        \{dot}
    $|        let s = s_dot * SCALE;
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
    $|  if (i < SQ) {
    $|    let inv = 1.0 / l_run;
    $|    \{store}
    $|  }
    $|}
  src
}

///|
/// Naive 4D attention baseline (benchmark reference): stage 1 materializes
/// the full Sq x Skv score matrix per (batch, head) slice into global
/// memory. One workgroup per 64 query rows, one thread per row, q row
/// unrolled into registers. Layout: scores[bh * SQ * SKV + i * SKV + j].
pub fn attn_naive_scores_shader(
  b : Int,
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  d : Int,
  causal : Bool,
  scale : Double,
) -> String {
  let qload = unroll_dv(d, fn(buf, t) {
    buf.write_string("qr[\{t}] = q[qoff + i * D + \{t}u];")
  })
  let dot = dot4(d, "qr", "kc[koff + j * D + ")
  let src =
    $|@group(0) @binding(0) var<storage, read> q : array<f32>;
    $|@group(0) @binding(1) var<storage, read> kc : array<f32>;
    $|@group(0) @binding(2) var<storage, read_write> sc : array<f32>;
    $|
    $|const B : u32 = \{b}u;
    $|const H : u32 = \{h}u;
    $|const HKV : u32 = \{hkv}u;
    $|const GROUP : u32 = \{h / hkv}u;
    $|const SQ : u32 = \{sq}u;
    $|const SKV : u32 = \{skv}u;
    $|const D : u32 = \{d}u;
    $|const DELTA : u32 = \{skv - sq}u;
    $|const CAUSAL : bool = \{causal};
    $|const SCALE : f32 = \{scale};
    $|const WG : u32 = 64u;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let bh = wid.y;
    $|  let bat = bh / H;
    $|  let head = bh % H;
    $|  let kvh = head / GROUP;
    $|  let i = wid.x * WG + lid.x;
    $|  if (i >= SQ) { return; }
    $|  let qoff = bh * SQ * D;
    $|  let koff = (bat * HKV + kvh) * SKV * D;
    $|  var mlim = SKV;
    $|  if (CAUSAL) {
    $|    mlim = min(i + DELTA + 1u, SKV);
    $|  }
    $|  var qr : array<f32, D>;
    $|  \{qload}
    $|  let sbase = bh * SQ * SKV + i * SKV;
    $|  for (var j = 0u; j < mlim; j = j + 1u) {
    $|    \{dot}
    $|    sc[sbase + j] = s_dot * SCALE;
    $|  }
    $|}
  src
}

///|
/// Naive 4D attention baseline, stage 2: row-wise softmax over the
/// materialized score matrix (read back from global memory) followed by
/// PV accumulation reading V straight from global memory. One thread per
/// query row; DV accumulation unrolled into registers.
pub fn attn_naive_pv_shader(
  h : Int,
  hkv : Int,
  sq : Int,
  skv : Int,
  dv : Int,
  causal : Bool,
) -> String {
  let acc = unroll_dv(dv, fn(buf, c) {
    buf.write_string("o[\{c}] = o[\{c}] + p * vc[voff + j * DV + \{c}u];")
  })
  let store = unroll_dv(dv, fn(buf, c) {
    buf.write_string("outp[(bh * SQ + i) * DV + \{c}u] = o[\{c}] * inv;")
  })
  let src =
    $|@group(0) @binding(0) var<storage, read> sc : array<f32>;
    $|@group(0) @binding(1) var<storage, read> vc : array<f32>;
    $|@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
    $|
    $|const H : u32 = \{h}u;
    $|const HKV : u32 = \{hkv}u;
    $|const GROUP : u32 = \{h / hkv}u;
    $|const SQ : u32 = \{sq}u;
    $|const SKV : u32 = \{skv}u;
    $|const DV : u32 = \{dv}u;
    $|const DELTA : u32 = \{skv - sq}u;
    $|const CAUSAL : bool = \{causal};
    $|const WG : u32 = 64u;
    $|const NEG_BIG : f32 = -3.402823466e+38;
    $|
    $|@compute @workgroup_size(64)
    $|fn main(
    $|  @builtin(workgroup_id) wid : vec3<u32>,
    $|  @builtin(local_invocation_id) lid : vec3<u32>,
    $|) {
    $|  let bh = wid.y;
    $|  let bat = bh / H;
    $|  let head = bh % H;
    $|  let kvh = head / GROUP;
    $|  let i = wid.x * WG + lid.x;
    $|  if (i >= SQ) { return; }
    $|  let voff = (bat * HKV + kvh) * SKV * DV;
    $|  let sbase = bh * SQ * SKV + i * SKV;
    $|  var mlim = SKV;
    $|  if (CAUSAL) {
    $|    mlim = min(i + DELTA + 1u, SKV);
    $|  }
    $|  var m = NEG_BIG;
    $|  for (var j = 0u; j < mlim; j = j + 1u) {
    $|    m = max(m, sc[sbase + j]);
    $|  }
    $|  var l = 0.0;
    $|  for (var j = 0u; j < mlim; j = j + 1u) {
    $|    l = l + exp(sc[sbase + j] - m);
    $|  }
    $|  let inv = 1.0 / l;
    $|  var o : array<f32, DV>;
    $|  for (var j = 0u; j < mlim; j = j + 1u) {
    $|    let p = exp(sc[sbase + j] - m);
    $|    \{acc}
    $|  }
    $|  \{store}
    $|}
  src
}
