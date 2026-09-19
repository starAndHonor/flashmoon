///| FlashAttention forward pass for MoonBit, built on moonxi-net's NpArray.
///
/// Batched multi-head scaled dot-product attention over 4D row-major
/// tensors: q [B, H, Sq, D], k [B, HKV, Skv, D], v [B, HKV, Skv, DV]
///   -> out [B, H, Sq, DV], out = softmax(q · kᵀ · scale) · v
/// per (batch, head) slice. HKV < H gives GQA (H % HKV == 0); HKV == 1
/// gives MQA. `causal` applies a bottom-right-aligned causal mask
/// (row i attends keys 0 ..= i + Skv - Sq), which covers self-attention
/// (Sq == Skv) and cached/cross attention (Sq < Skv).
///
/// Hot loops run on little-endian f32 byte buffers through `kdot` / `kaxpy`
/// / `kscale`, which are wasm SIMD (f32x4) kernels on the wasm target and
/// scalar fallbacks elsewhere (see kernel_wasm.mbt / kernel_scalar.mbt).
/// Both handle arbitrary row lengths (SIMD kernels have a scalar tail),
/// so any head dim works on every target.

///|
/// Attention configuration shared by `naive_attention` / `flash_attention`
/// (and the WebGPU backend in `flash/gpu`).
pub struct AttnConfig {
  /// Query heads H.
  heads : Int
  /// Key/value heads HKV (H % HKV must be 0; HKV == H is MHA, 1 is MQA).
  kv_heads : Int
  /// Bottom-right-aligned causal mask.
  causal : Bool
  /// Score scale; `None` uses 1/sqrt(D).
  scale : Float?
}

///|
pub fn AttnConfig::new(
  heads~ : Int,
  kv_heads? : Int = heads,
  causal? : Bool = false,
  scale? : Float? = None,
) -> AttnConfig {
  if heads <= 0 || kv_heads <= 0 {
    abort("heads must be positive, got heads=\{heads} kv_heads=\{kv_heads}")
  }
  if heads % kv_heads != 0 {
    abort("heads % kv_heads != 0: \{heads} vs \{kv_heads}")
  }
  { heads, kv_heads, causal, scale }
}

///|
/// Extracted, validated problem shape.
priv struct Shape {
  qd : FixedArray[Float]
  kd : FixedArray[Float]
  vd : FixedArray[Float]
  b : Int
  h : Int
  sq : Int
  skv : Int
  d : Int
  dv : Int
}

///|
/// Validate q/k/v as 4D [B,H,Sq,D] / [B,HKV,Skv,D] / [B,HKV,Skv,DV]
/// against `cfg` and return the flat buffers + shape. Aborts on mismatch.
fn extract_qkv(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  cfg : AttnConfig,
) -> Shape {
  let qt = q.value()
  let kt = k.value()
  let vt = v.value()
  if qt.dims.length() != 4 || kt.dims.length() != 4 || vt.dims.length() != 4 {
    abort("expected 4D tensors [B,H,S,D] for q/k/v")
  }
  let b = qt.dims[0]
  let h = qt.dims[1]
  let sq = qt.dims[2]
  let d = qt.dims[3]
  if h != cfg.heads {
    abort("q heads \{h} != config heads \{cfg.heads}")
  }
  if kt.dims[0] != b || vt.dims[0] != b {
    abort("batch mismatch: q \{b}, k \{kt.dims[0]}, v \{vt.dims[0]}")
  }
  if kt.dims[1] != cfg.kv_heads || vt.dims[1] != cfg.kv_heads {
    abort(
      "kv heads mismatch: k \{kt.dims[1]}, v \{vt.dims[1]}, config \{cfg.kv_heads}",
    )
  }
  let skv = kt.dims[2]
  if kt.dims[3] != d {
    abort("q/k head-dim mismatch: \{d} vs \{kt.dims[3]}")
  }
  if vt.dims[2] != skv {
    abort("k/v seq-len mismatch: \{skv} vs \{vt.dims[2]}")
  }
  if cfg.causal && skv < sq {
    abort("causal requires Skv >= Sq, got Skv=\{skv} Sq=\{sq}")
  }
  { qd: qt.data, kd: kt.data, vd: vt.data, b, h, sq, skv, d, dv: vt.dims[3] }
}

///|
/// Convert a byte-buffer row block [rows, dv] at element base 0 of `acc`
/// into `out` floats at `outoff + r*dv`.
fn copy_rows_out(
  acc : FixedArray[Byte],
  out : FixedArray[Float],
  rows : Int,
  dv : Int,
  outoff : Int,
) -> Unit {
  for r in 0..<rows {
    let abase = r * dv
    let obase = outoff + r * dv
    for c in 0..<dv {
      out.unsafe_set(obase + c, get_f32(acc, abase + c))
    }
  }
}

///|
/// Key count visible to query row `i`: bottom-right-aligned causal limit
/// (delta = Skv - Sq >= 0), or all Skv keys when not causal.
fn key_limit(causal : Bool, i : Int, m : Int, delta : Int) -> Int {
  if causal {
    let lim = i + delta + 1
    if lim < m {
      lim
    } else {
      m
    }
  } else {
    m
  }
}

///|
/// Naive single-slice attention on little-endian f32 byte buffers with
/// base offsets: materializes the row score vector, applies softmax,
/// multiplies by v. The reference oracle for `flash_attention` and the
/// WebGPU backend.
fn naive2d(
  qbytes : FixedArray[Byte],
  kbytes : FixedArray[Byte],
  vbytes : FixedArray[Byte],
  qoff : Int,
  koff : Int,
  voff : Int,
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
  scale : Float,
  causal : Bool,
  out : FixedArray[Float],
  outoff : Int,
) -> Unit {
  let acc = FixedArray::make(n * dv * 4, b'\x00')
  let scores = FixedArray::make(m, 0.0F)
  let delta = m - n
  for i in 0..<n {
    let lim = key_limit(causal, i, m, delta)
    let qbase = qoff + i * d
    let mut row_max = @float.neg_infinity
    for j in 0..<lim {
      let sj = kdot(qbytes, qbase, kbytes, koff + j * d, d) * scale
      scores.unsafe_set(j, sj)
      if sj > row_max {
        row_max = sj
      }
    }
    let mut row_sum = 0.0F
    for j in 0..<lim {
      let p = @math.expf(scores.unsafe_get(j) - row_max)
      scores.unsafe_set(j, p)
      row_sum += p
    }
    for j in 0..<lim {
      kaxpy(acc, i * dv, scores.unsafe_get(j), vbytes, voff + j * dv, dv)
    }
    kscale(acc, i * dv, 1.0F / row_sum, dv)
  }
  copy_rows_out(acc, out, n, dv, outoff)
}

///|
/// Flash single-slice attention: tiles q rows and k/v columns into blocks
/// and folds each k/v block into the output with an online softmax update
/// (running row max `m_run`, running normalizer `l_run`, rescaled
/// accumulator `acc`). Never materializes the full [n, m] scores.
fn flash2d(
  qbytes : FixedArray[Byte],
  kbytes : FixedArray[Byte],
  vbytes : FixedArray[Byte],
  qoff : Int,
  koff : Int,
  voff : Int,
  n : Int,
  m : Int,
  d : Int,
  dv : Int,
  scale : Float,
  causal : Bool,
  block_rows : Int,
  block_cols : Int,
  out : FixedArray[Float],
  outoff : Int,
) -> Unit {
  let s_blk = FixedArray::make(block_rows * block_cols, 0.0F)
  let m_run = FixedArray::make(block_rows, 0.0F)
  let l_run = FixedArray::make(block_rows, 0.0F)
  let acc = FixedArray::make(block_rows * dv * 4, b'\x00')
  let delta = m - n
  let n_row_blocks = (n + block_rows - 1) / block_rows
  for bi in 0..<n_row_blocks {
    let i0 = bi * block_rows
    let rows = if block_rows < n - i0 { block_rows } else { n - i0 }
    m_run.fill(@float.neg_infinity)
    l_run.fill(0.0F)
    kscale(acc, 0, 0.0F, rows * dv)
    // deepest key bound among this block's rows
    let m_blk = key_limit(causal, i0 + rows - 1, m, delta)
    let n_col_blocks = (m_blk + block_cols - 1) / block_cols
    for bj in 0..<n_col_blocks {
      let j0 = bj * block_cols
      let cols = if block_cols < m_blk - j0 { block_cols } else { m_blk - j0 }
      // S = q_blk · k_blkᵀ · scale, clipped per row at its causal limit
      for r in 0..<rows {
        let qbase = qoff + (i0 + r) * d
        let sbase = r * block_cols
        let lim = key_limit(causal, i0 + r, m, delta) - j0
        let rcols = if cols < lim { cols } else { lim }
        for c in 0..<rcols {
          s_blk.unsafe_set(
            sbase + c,
            kdot(qbytes, qbase, kbytes, koff + (j0 + c) * d, d) * scale,
          )
        }
      }
      // Online softmax update per row of the block
      for r in 0..<rows {
        let sbase = r * block_cols
        let lim = key_limit(causal, i0 + r, m, delta) - j0
        let rcols = if cols < lim { cols } else { lim }
        let mut mx = m_run.unsafe_get(r)
        for c in 0..<rcols {
          let x = s_blk.unsafe_get(sbase + c)
          if x > mx {
            mx = x
          }
        }
        // exp(-inf) = 0 on the first block (m_run = -inf)
        let alpha = @math.expf(m_run.unsafe_get(r) - mx)
        let mut psum = 0.0F
        for c in 0..<rcols {
          let p = @math.expf(s_blk.unsafe_get(sbase + c) - mx)
          s_blk.unsafe_set(sbase + c, p)
          psum += p
        }
        l_run.unsafe_set(r, l_run.unsafe_get(r) * alpha + psum)
        // acc = alpha * acc + P_blk · v_blk
        let abase = r * dv
        kscale(acc, abase, alpha, dv)
        for j in 0..<rcols {
          kaxpy(
            acc,
            abase,
            s_blk.unsafe_get(sbase + j),
            vbytes,
            voff + (j0 + j) * dv,
            dv,
          )
        }
        m_run.unsafe_set(r, mx)
      }
    }
    for r in 0..<rows {
      kscale(acc, r * dv, 1.0F / l_run.unsafe_get(r), dv)
    }
    copy_rows_out(acc, out, rows, dv, outoff + i0 * dv)
  }
}

///|
/// Invoke `core` once per (batch, head) slice with the flat base offsets
/// (qoff, koff, voff, outoff), resolving the GQA head mapping.
fn run_slices(
  s : Shape,
  cfg : AttnConfig,
  core : (Int, Int, Int, Int) -> Unit,
) -> Unit {
  let group = s.h / cfg.kv_heads
  for bi in 0..<s.b {
    for hi in 0..<s.h {
      let kvh = hi / group
      let qoff = (bi * s.h + hi) * s.sq * s.d
      let koff = (bi * cfg.kv_heads + kvh) * s.skv * s.d
      let voff = (bi * cfg.kv_heads + kvh) * s.skv * s.dv
      let outoff = (bi * s.h + hi) * s.sq * s.dv
      core(qoff, koff, voff, outoff)
    }
  }
}

///|
/// Reference multi-head attention: materializes each row's full score
/// vector. Simple, obviously correct, and the oracle for `flash_attention`.
pub fn naive_attention(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  cfg : AttnConfig,
) -> @nparray.NpArray {
  let s = extract_qkv(q, k, v, cfg)
  let scale = match cfg.scale {
    Some(x) => x
    None => 1.0F / Float::from_int(s.d).sqrt()
  }
  let out = FixedArray::make(s.b * s.h * s.sq * s.dv, 0.0F)
  let qbytes = floats_to_bytes(s.qd)
  let kbytes = floats_to_bytes(s.kd)
  let vbytes = floats_to_bytes(s.vd)
  run_slices(s, cfg, fn(qoff, koff, voff, outoff) {
    naive2d(
      qbytes,
      kbytes,
      vbytes,
      qoff,
      koff,
      voff,
      s.sq,
      s.skv,
      s.d,
      s.dv,
      scale,
      cfg.causal,
      out,
      outoff,
    )
  })
  @tensor.Tensor::from_host(out, [s.b, s.h, s.sq, s.dv])
}

///|
/// FlashAttention forward: per (batch, head) slice, tiles q rows and k/v
/// columns and folds tiles into the output with an online softmax update.
/// Same result as `naive_attention` up to float summation order, without
/// materializing the [Sq, Skv] score matrix. `block_rows` / `block_cols`
/// control the q / kv tile sizes.
pub fn flash_attention(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  cfg : AttnConfig,
  block_rows? : Int = 64,
  block_cols? : Int = 64,
) -> @nparray.NpArray {
  if block_rows <= 0 || block_cols <= 0 {
    abort("block sizes must be positive")
  }
  let s = extract_qkv(q, k, v, cfg)
  let scale = match cfg.scale {
    Some(x) => x
    None => 1.0F / Float::from_int(s.d).sqrt()
  }
  let out = FixedArray::make(s.b * s.h * s.sq * s.dv, 0.0F)
  let qbytes = floats_to_bytes(s.qd)
  let kbytes = floats_to_bytes(s.kd)
  let vbytes = floats_to_bytes(s.vd)
  run_slices(s, cfg, fn(qoff, koff, voff, outoff) {
    flash2d(
      qbytes,
      kbytes,
      vbytes,
      qoff,
      koff,
      voff,
      s.sq,
      s.skv,
      s.d,
      s.dv,
      scale,
      cfg.causal,
      block_rows,
      block_cols,
      out,
      outoff,
    )
  })
  @tensor.Tensor::from_host(out, [s.b, s.h, s.sq, s.dv])
}
