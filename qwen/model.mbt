///|
/// Qwen3-0.6B CPU reference forward (pure f32, naive ops — correctness
/// oracle for the WebGPU kernels, not for speed).

///|
/// Row-major [out, in] weight matrix, as stored in HF checkpoints.
struct Mat {
  data : FixedArray[Float]
  out : Int
  inn : Int
}

///|
struct Layer {
  q : Mat
  k : Mat
  v : Mat
  o : Mat
  q_norm : FixedArray[Float] // [head_dim]
  k_norm : FixedArray[Float]
  in_norm : FixedArray[Float] // [hidden]
  post_norm : FixedArray[Float]
  gate : Mat
  up : Mat
  down : Mat
}

///|
pub struct Qwen3 {
  embed : Mat // [vocab, hidden]; lm_head tied to this
  layers : Array[Layer]
  norm : FixedArray[Float]
  n_layer : Int
  hidden : Int
  n_head : Int
  n_kv : Int
  head_dim : Int
  intermediate : Int
  eps : Float
  rope_theta : Double
  vocab : Int
}

///|
fn get_mat(st : SafeTensors, name : String) -> Mat raise Error {
  let (data, shape) = st.get_f32(name)
  guard shape.length() == 2 else { raise Failure("\{name}: expected 2D") }
  { data, out: shape[0], inn: shape[1] }
}

///|
fn get_vec(st : SafeTensors, name : String) -> FixedArray[Float] raise Error {
  let (data, shape) = st.get_f32(name)
  guard shape.length() == 1 else { raise Failure("\{name}: expected 1D") }
  data
}

///|
/// Load from a parsed SafeTensors. Model config is the released Qwen3-0.6B:
/// hidden 1024, 28 layers, 16 q heads / 8 kv heads, head_dim 128,
/// intermediate 3072, tied embeddings, rope_theta 1e6, rms eps 1e-6.
pub fn Qwen3::load(st : SafeTensors) -> Qwen3 raise Error {
  let layers : Array[Layer] = []
  for i in 0..<28 {
    let p = "model.layers.\{i}"
    layers.push({
      q: get_mat(st, "\{p}.self_attn.q_proj.weight"),
      k: get_mat(st, "\{p}.self_attn.k_proj.weight"),
      v: get_mat(st, "\{p}.self_attn.v_proj.weight"),
      o: get_mat(st, "\{p}.self_attn.o_proj.weight"),
      q_norm: get_vec(st, "\{p}.self_attn.q_norm.weight"),
      k_norm: get_vec(st, "\{p}.self_attn.k_norm.weight"),
      in_norm: get_vec(st, "\{p}.input_layernorm.weight"),
      post_norm: get_vec(st, "\{p}.post_attention_layernorm.weight"),
      gate: get_mat(st, "\{p}.mlp.gate_proj.weight"),
      up: get_mat(st, "\{p}.mlp.up_proj.weight"),
      down: get_mat(st, "\{p}.mlp.down_proj.weight"),
    })
  }
  {
    embed: get_mat(st, "model.embed_tokens.weight"),
    layers,
    norm: get_vec(st, "model.norm.weight"),
    n_layer: 28,
    hidden: 1024,
    n_head: 16,
    n_kv: 8,
    head_dim: 128,
    intermediate: 3072,
    eps: 1.0e-6,
    rope_theta: 1000000.0,
    vocab: 151936,
  }
}

///|
/// y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
fn rmsnorm(
  x : FixedArray[Float],
  w : FixedArray[Float],
  eps : Float,
  y : FixedArray[Float],
) -> Unit {
  let n = x.length()
  let mut ss = 0.0
  for i in 0..<n {
    let v = x[i].to_double()
    ss += v * v
  }
  let inv = Float::from_double(
    1.0 / (ss / n.to_double() + eps.to_double()).sqrt(),
  )
  for i in 0..<n {
    y[i] = x[i] * inv * w[i]
  }
}

///|
/// y[o] = sum_i w[o, i] * x[i] — matvec with row-major [out, in] weight.
fn matvec(w : Mat, x : FixedArray[Float], y : FixedArray[Float]) -> Unit {
  for o in 0..<w.out {
    let base = o * w.inn
    let mut acc : Float = 0.0
    for i in 0..<w.inn {
      acc += w.data[base + i] * x[i]
    }
    y[o] = acc
  }
}

///|
/// NeoX-style RoPE over one head vector (in place): pairs (i, i + dim/2).
fn rope(
  vec : FixedArray[Float],
  off : Int,
  dim : Int,
  pos : Int,
  theta : Double,
) -> Unit {
  let half = dim / 2
  for i in 0..<half {
    let freq = @math.pow(theta, -2.0 * i.to_double() / dim.to_double()) *
      pos.to_double()
    let c = Float::from_double(@math.cos(freq))
    let s = Float::from_double(@math.sin(freq))
    let x1 = vec[off + i]
    let x2 = vec[off + i + half]
    vec[off + i] = x1 * c - x2 * s
    vec[off + i + half] = x1 * s + x2 * c
  }
}

///|
fn silu(x : Float) -> Float {
  let e = Float::from_double(@math.exp(-x.to_double()))
  x / (Float::from_double(1.0) + e)
}

///|
/// Full-prompt forward with causal masking; returns last-position logits
/// [vocab]. Naive attention over all positions — M0 oracle only.
pub fn Qwen3::forward(self : Qwen3, tokens : Array[Int]) -> FixedArray[Float] {
  let s = tokens.length()
  let hd = self.head_dim
  let h = self.hidden
  // x: [s, hidden] activations
  let x : FixedArray[Float] = FixedArray::make(s * h, (0.0 : Float))
  for t in 0..<s {
    let row = tokens[t] * h
    for i in 0..<h {
      x[t * h + i] = self.embed.data[row + i]
    }
  }
  let norm_buf : FixedArray[Float] = FixedArray::make(h, (0.0 : Float))
  let qq : FixedArray[Float] = FixedArray::make(self.n_head * hd, (0.0 : Float))
  let kk : FixedArray[Float] = FixedArray::make(self.n_kv * hd, (0.0 : Float))
  let vv : FixedArray[Float] = FixedArray::make(self.n_kv * hd, (0.0 : Float))
  let attn_out : FixedArray[Float] = FixedArray::make(
    self.n_head * hd,
    (0.0 : Float),
  )
  let proj_out : FixedArray[Float] = FixedArray::make(h, (0.0 : Float))
  let gate_b : FixedArray[Float] = FixedArray::make(
    self.intermediate,
    (0.0 : Float),
  )
  let up_b : FixedArray[Float] = FixedArray::make(
    self.intermediate,
    (0.0 : Float),
  )
  let mlp_out : FixedArray[Float] = FixedArray::make(h, (0.0 : Float))
  // KV cache: [layer][kv_head][pos][dim]
  let kcache : Array[FixedArray[Float]] = []
  let vcache : Array[FixedArray[Float]] = []
  for _ in 0..<self.n_layer {
    kcache.push(FixedArray::make(s * self.n_kv * hd, (0.0 : Float)))
    vcache.push(FixedArray::make(s * self.n_kv * hd, (0.0 : Float)))
  }
  let scores : FixedArray[Float] = FixedArray::make(s, (0.0 : Float))
  let scale = Float::from_double(1.0 / hd.to_double().sqrt())
  for li in 0..<self.n_layer {
    let layer = self.layers[li]
    let kc = kcache[li]
    let vc = vcache[li]
    for t in 0..<s {
      let xt = FixedArray::make(h, (0.0 : Float))
      for i in 0..<h {
        xt[i] = x[t * h + i]
      }
      rmsnorm(xt, layer.in_norm, self.eps, norm_buf)
      matvec(layer.q, norm_buf, qq)
      matvec(layer.k, norm_buf, kk)
      matvec(layer.v, norm_buf, vv)
      // QK-norm per head, then RoPE
      let hd_vec : FixedArray[Float] = FixedArray::make(hd, (0.0 : Float))
      let hd_out : FixedArray[Float] = FixedArray::make(hd, (0.0 : Float))
      for head in 0..<self.n_head {
        for i in 0..<hd {
          hd_vec[i] = qq[head * hd + i]
        }
        rmsnorm(hd_vec, layer.q_norm, self.eps, hd_out)
        for i in 0..<hd {
          qq[head * hd + i] = hd_out[i]
        }
        rope(qq, head * hd, hd, t, self.rope_theta)
      }
      for head in 0..<self.n_kv {
        for i in 0..<hd {
          hd_vec[i] = kk[head * hd + i]
        }
        rmsnorm(hd_vec, layer.k_norm, self.eps, hd_out)
        for i in 0..<hd {
          kk[head * hd + i] = hd_out[i]
        }
        rope(kk, head * hd, hd, t, self.rope_theta)
      }
      // cache k/v
      for head in 0..<self.n_kv {
        for i in 0..<hd {
          kc[(head * s + t) * hd + i] = kk[head * hd + i]
          vc[(head * s + t) * hd + i] = vv[head * hd + i]
        }
      }
      // causal attention, GQA: q head g maps to kv head g / (n_head/n_kv)
      let group = self.n_head / self.n_kv
      for head in 0..<self.n_head {
        let kvh = head / group
        let mut mx : Float = -3.402823466e+38
        for j in 0..<=t {
          let mut acc : Float = 0.0
          for i in 0..<hd {
            acc += qq[head * hd + i] * kc[(kvh * s + j) * hd + i]
          }
          scores[j] = acc * scale
          if scores[j] > mx {
            mx = scores[j]
          }
        }
        let mut sum : Float = 0.0
        for j in 0..<=t {
          let e = Float::from_double(@math.exp((scores[j] - mx).to_double()))
          scores[j] = e
          sum += e
        }
        for i in 0..<hd {
          let mut acc : Float = 0.0
          for j in 0..<=t {
            acc += scores[j] * vc[(kvh * s + j) * hd + i]
          }
          attn_out[head * hd + i] = acc / sum
        }
      }
      matvec(layer.o, attn_out, proj_out)
      for i in 0..<h {
        x[t * h + i] = xt[i] + proj_out[i]
      }
      // MLP
      for i in 0..<h {
        xt[i] = x[t * h + i]
      }
      rmsnorm(xt, layer.post_norm, self.eps, norm_buf)
      matvec(layer.gate, norm_buf, gate_b)
      matvec(layer.up, norm_buf, up_b)
      for i in 0..<self.intermediate {
        gate_b[i] = silu(gate_b[i]) * up_b[i]
      }
      matvec(layer.down, gate_b, mlp_out)
      for i in 0..<h {
        x[t * h + i] = xt[i] + mlp_out[i]
      }
    }
  }
  // final norm on last position, tied LM head
  let last = (s - 1) * h
  let xt : FixedArray[Float] = FixedArray::make(h, (0.0 : Float))
  for i in 0..<h {
    xt[i] = x[last + i]
  }
  rmsnorm(xt, self.norm, self.eps, norm_buf)
  let logits : FixedArray[Float] = FixedArray::make(self.vocab, (0.0 : Float))
  matvec(self.embed, norm_buf, logits)
  logits
}
