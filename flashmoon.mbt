///| FlashAttention forward pass for MoonBit, built on moonxi-net's NpArray.
///
/// Single-head scaled dot-product attention over 2D row-major tensors:
///   q: [n, d], k: [m, d], v: [m, dv]  →  out: [n, dv]
///   out = softmax(q · kᵀ / √d) · v
///
/// Hot loops run on little-endian f32 byte buffers through `kdot` / `kaxpy`
/// / `kscale`, which are wasm SIMD (f32x4) kernels on the wasm target and
/// scalar fallbacks elsewhere (see kernel_wasm.mbt / kernel_scalar.mbt).

///|
/// Extract row-major data and 2D shape; aborts on non-2D input.
fn expect2d(
  t : @nparray.NpArray,
  name : String,
) -> (FixedArray[Float], Int, Int) {
  let td = t.value()
  if td.dims.length() != 2 {
    abort("\{name}: expected 2D tensor, got dims.length()=\{td.dims.length()}")
  }
  (td.data, td.dims[0], td.dims[1])
}

///|
/// Extract q/k/v shapes and validate: q [n,d], k [m,d], v [m,dv].
fn extract_qkv(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
) -> (
  FixedArray[Float],
  FixedArray[Float],
  FixedArray[Float],
  Int,
  Int,
  Int,
  Int,
) {
  let (qd, n, d) = expect2d(q, "q")
  let (kd, m, dk) = expect2d(k, "k")
  let (vd, vm, dv) = expect2d(v, "v")
  if d != dk {
    abort("q/k head-dim mismatch: \{d} vs \{dk}")
  }
  if m != vm {
    abort("k/v seq-len mismatch: \{m} vs \{vm}")
  }
  (qd, kd, vd, n, m, d, dv)
}

///|
/// Convert a byte-buffer row block [rows, dv] at element base `src_base`
/// into a fresh float tensor row-major [rows, dv]... into `out` at `dst_base`.
fn copy_rows_out(
  acc : FixedArray[Byte],
  out : FixedArray[Float],
  rows : Int,
  dv : Int,
  i0 : Int,
) -> Unit {
  for r in 0..<rows {
    let abase = r * dv
    let obase = (i0 + r) * dv
    for c in 0..<dv {
      out.unsafe_set(obase + c, get_f32(acc, abase + c))
    }
  }
}

///|
/// Naive reference attention: materializes the full [n, m] score matrix,
/// applies row softmax, then multiplies by v.
pub fn naive_attention(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
) -> @nparray.NpArray {
  let (qd, kd, vd, n, m, d, dv) = extract_qkv(q, k, v)
  let scale = 1.0F / Float::from_int(d).sqrt()
  let qbytes = floats_to_bytes(qd)
  let kbytes = floats_to_bytes(kd)
  let vbytes = floats_to_bytes(vd)
  let out = FixedArray::make(n * dv * 4, b'\x00')
  let scores = FixedArray::make(m, 0.0F)
  for i in 0..<n {
    // S_i = q_i · kᵀ · scale, tracking the row max for stability
    let qbase = i * d
    let mut row_max = @float.neg_infinity
    for j in 0..<m {
      let sj = kdot(qbytes, qbase, kbytes, j * d, d) * scale
      scores.unsafe_set(j, sj)
      if sj > row_max {
        row_max = sj
      }
    }
    // softmax in place: scores[j] = exp(sj - max), normalized below
    let mut row_sum = 0.0F
    for j in 0..<m {
      let p = @math.expf(scores.unsafe_get(j) - row_max)
      scores.unsafe_set(j, p)
      row_sum += p
    }
    // out_i = P_i · v (accumulate), then divide by the softmax sum
    for j in 0..<m {
      kaxpy(out, i * dv, scores.unsafe_get(j), vbytes, j * dv, dv)
    }
    kscale(out, i * dv, 1.0F / row_sum, dv)
  }
  let out_f = FixedArray::make(n * dv, 0.0F)
  copy_rows_out(out, out_f, n, dv, 0)
  @tensor.Tensor::from_host(out_f, [n, dv])
}

///|
/// FlashAttention forward: tiles q rows and k/v columns into blocks and
/// folds each k/v block into the output with an online softmax update
/// (running row max `m_run`, running normalizer `l_run`, rescaled
/// accumulator `acc`). Never materializes the full [n, m] scores.
///
/// `block_rows` / `block_cols` control the q / kv tile sizes.
pub fn flash_attention(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  block_rows? : Int = 64,
  block_cols? : Int = 64,
) -> @nparray.NpArray {
  let (qd, kd, vd, n, m, d, dv) = extract_qkv(q, k, v)
  if block_rows <= 0 || block_cols <= 0 {
    abort("block sizes must be positive")
  }
  let scale = 1.0F / Float::from_int(d).sqrt()
  let qbytes = floats_to_bytes(qd)
  let kbytes = floats_to_bytes(kd)
  let vbytes = floats_to_bytes(vd)
  // Per-block scratch buffers
  let s_blk = FixedArray::make(block_rows * block_cols, 0.0F)
  let m_run = FixedArray::make(block_rows, 0.0F)
  let l_run = FixedArray::make(block_rows, 0.0F)
  let acc = FixedArray::make(block_rows * dv * 4, b'\x00')
  let out = FixedArray::make(n * dv, 0.0F)
  let n_row_blocks = (n + block_rows - 1) / block_rows
  let n_col_blocks = (m + block_cols - 1) / block_cols
  for bi in 0..<n_row_blocks {
    let i0 = bi * block_rows
    let rows = if block_rows < n - i0 { block_rows } else { n - i0 }
    m_run.fill(@float.neg_infinity)
    l_run.fill(0.0F)
    kscale(acc, 0, 0.0F, rows * dv) // reset accumulator
    for bj in 0..<n_col_blocks {
      let j0 = bj * block_cols
      let cols = if block_cols < m - j0 { block_cols } else { m - j0 }
      // S = q_blk · k_blkᵀ · scale  (rows × cols)
      for r in 0..<rows {
        let qbase = (i0 + r) * d
        let sbase = r * block_cols
        for c in 0..<cols {
          s_blk.unsafe_set(
            sbase + c,
            kdot(qbytes, qbase, kbytes, (j0 + c) * d, d) * scale,
          )
        }
      }
      // Online softmax update per row of the block
      for r in 0..<rows {
        let sbase = r * block_cols
        let mut mx = m_run.unsafe_get(r)
        for c in 0..<cols {
          let x = s_blk.unsafe_get(sbase + c)
          if x > mx {
            mx = x
          }
        }
        // Rescale factor for the running state; exp(-inf) = 0 on the first block
        let alpha = @math.expf(m_run.unsafe_get(r) - mx)
        let mut psum = 0.0F
        for c in 0..<cols {
          let p = @math.expf(s_blk.unsafe_get(sbase + c) - mx)
          s_blk.unsafe_set(sbase + c, p)
          psum += p
        }
        l_run.unsafe_set(r, l_run.unsafe_get(r) * alpha + psum)
        // acc = alpha * acc + P_blk · v_blk
        let abase = r * dv
        kscale(acc, abase, alpha, dv)
        for j in 0..<cols {
          kaxpy(
            acc,
            abase,
            s_blk.unsafe_get(sbase + j),
            vbytes,
            (j0 + j) * dv,
            dv,
          )
        }
        m_run.unsafe_set(r, mx)
      }
    }
    // Normalize by the softmax denominator, then copy rows out
    for r in 0..<rows {
      kscale(acc, r * dv, 1.0F / l_run.unsafe_get(r), dv)
    }
    copy_rows_out(acc, out, rows, dv, i0)
  }
  @tensor.Tensor::from_host(out, [n, dv])
}
