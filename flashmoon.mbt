///|
/// FlashAttention forward pass for MoonBit, built on moonxi-net's NpArray.
///
/// Single-head scaled dot-product attention over 2D row-major tensors:
///   q: [n, d], k: [m, d], v: [m, dv]  →  out: [n, dv]
///   out = softmax(q · kᵀ / √d) · v

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
/// Naive reference attention: materializes the full [n, m] score matrix,
/// applies row softmax, then multiplies by v.
pub fn naive_attention(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
) -> @nparray.NpArray {
  let (qd, kd, vd, n, m, d, dv) = extract_qkv(q, k, v)
  let scale = 1.0F / Float::from_int(d).sqrt()
  let out = FixedArray::make(n * dv, 0.0F)
  let scores = FixedArray::make(m, 0.0F)
  for i in 0..<n {
    // S_i = q_i · kᵀ · scale, tracking the row max for stability
    let mut row_max = @float.neg_infinity
    for j in 0..<m {
      let mut s = 0.0F
      for t in 0..<d {
        s += qd[i * d + t] * kd[j * d + t]
      }
      let sj = s * scale
      scores[j] = sj
      if sj > row_max {
        row_max = sj
      }
    }
    // softmax in place: scores[j] = exp(sj - max), normalized below
    let mut row_sum = 0.0F
    for j in 0..<m {
      let p = @math.expf(scores[j] - row_max)
      scores[j] = p
      row_sum += p
    }
    // out_i = (P_i · v) / sum
    for c in 0..<dv {
      let mut acc = 0.0F
      for j in 0..<m {
        acc += scores[j] * vd[j * dv + c]
      }
      out[i * dv + c] = acc / row_sum
    }
  }
  @tensor.Tensor::from_host(out, [n, dv])
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
  block_rows? : Int = 16,
  block_cols? : Int = 16,
) -> @nparray.NpArray {
  let (qd, kd, vd, n, m, d, dv) = extract_qkv(q, k, v)
  if block_rows <= 0 || block_cols <= 0 {
    abort("block sizes must be positive")
  }
  let scale = 1.0F / Float::from_int(d).sqrt()
  let out = FixedArray::make(n * dv, 0.0F)
  // Per-block scratch buffers
  let s_blk = FixedArray::make(block_rows * block_cols, 0.0F)
  let m_run = FixedArray::make(block_rows, 0.0F)
  let l_run = FixedArray::make(block_rows, 0.0F)
  let acc = FixedArray::make(block_rows * dv, 0.0F)
  let n_row_blocks = (n + block_rows - 1) / block_rows
  let n_col_blocks = (m + block_cols - 1) / block_cols
  for bi in 0..<n_row_blocks {
    let i0 = bi * block_rows
    let rows = if block_rows < n - i0 { block_rows } else { n - i0 }
    m_run.fill(@float.neg_infinity)
    l_run.fill(0.0F)
    acc.fill(0.0F)
    for bj in 0..<n_col_blocks {
      let j0 = bj * block_cols
      let cols = if block_cols < m - j0 { block_cols } else { m - j0 }
      // S = q_blk · k_blkᵀ · scale  (rows × cols)
      for r in 0..<rows {
        for c in 0..<cols {
          let mut sum = 0.0F
          for t in 0..<d {
            sum += qd[(i0 + r) * d + t] * kd[(j0 + c) * d + t]
          }
          s_blk[r * block_cols + c] = sum * scale
        }
      }
      // Online softmax update per row of the block
      for r in 0..<rows {
        let mut mx = m_run[r]
        for c in 0..<cols {
          let x = s_blk[r * block_cols + c]
          if x > mx {
            mx = x
          }
        }
        // Rescale factor for the running state; exp(-inf) = 0 on the first block
        let alpha = @math.expf(m_run[r] - mx)
        let mut psum = 0.0F
        for c in 0..<cols {
          let p = @math.expf(s_blk[r * block_cols + c] - mx)
          s_blk[r * block_cols + c] = p
          psum += p
        }
        l_run[r] = l_run[r] * alpha + psum
        // acc = alpha * acc + P_blk · v_blk
        for c in 0..<dv {
          let mut o = acc[r * dv + c] * alpha
          for j in 0..<cols {
            o += s_blk[r * block_cols + j] * vd[(j0 + j) * dv + c]
          }
          acc[r * dv + c] = o
        }
        m_run[r] = mx
      }
    }
    // Normalize by the softmax denominator
    for r in 0..<rows {
      let inv = 1.0F / l_run[r]
      for c in 0..<dv {
        out[(i0 + r) * dv + c] = acc[r * dv + c] * inv
      }
    }
  }
  @tensor.Tensor::from_host(out, [n, dv])
}
