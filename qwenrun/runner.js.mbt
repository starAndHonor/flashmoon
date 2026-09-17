// SPDX-License-Identifier: Apache-2.0
// Copyright Text: 2026 starandhonor

///| Qwen3-0.6B WebGPU runner (js target) — host-agnostic core shared by the

///| Deno REPL (cmd/qwengpu) and the browser page (cmd/webchat). Hosts

///| provide file reading and logging as function values.

///|

///| NOTE: no compute dispatch may bind the same buffer at two binding

///| points (writable-storage aliasing is a Dawn/Chrome validation error —

///| wgpu/Deno silently tolerates it). All kernels are single-binding

///| in-place or out-of-place.

///|
extern "js" fn now_ms() -> Double =
  #|() => performance.now()

///|
extern "js" fn randf() -> Double =
  #|() => Math.random()

///|
extern "js" fn bytes_to_string(b : Bytes) -> String =
  #|(b) => new TextDecoder().decode(b)

///|
/// Decode a raw safetensors payload to a Float32Array without the js
/// backend's 2^27-element cap on MoonBit FixedArray (lm_head is 155M).
extern "js" fn st_convert(
  data : Bytes,
  off : Int,
  n : Int,
  dtype : String,
) -> @gpu.F32Buf =
  #|(data, off, n, dtype) => {
  #|  const out = new Float32Array(n);
  #|  if (dtype === "BF16") {
  #|    const u16 = new Uint16Array(data.buffer, data.byteOffset + off, n);
  #|    const u32 = new Uint32Array(out.buffer);
  #|    for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
  #|  } else {
  #|    out.set(new Float32Array(data.buffer, data.byteOffset + off, n));
  #|  }
  #|  return out;
  #|}

///|
const NL : Int = 28

///|
const HID : Int = 1024

///|
const NQH : Int = 16

///|
const NKH : Int = 8

///|
const HD : Int = 128

///|
const INTER : Int = 3072

///|
const VOCAB : Int = 151936

///|
const MAXPOS : Int = 2048

///|
const EPS : Float = 1.0e-6

///|
/// Per-layer weight buffers + KV caches, all GPU-resident.
struct Model {
  w : Map[String, @gpu.GPUBuffer]
  kc : Array[@gpu.GPUBuffer]
  vc : Array[@gpu.GPUBuffer]
}

///|
fn upload(
  g : @gpu.Gpu,
  st : @qwen3.SafeTensors,
  name : String,
) -> @gpu.GPUBuffer raise Error {
  let (bytes, off, n, dtype) = st.raw(name)
  g.upload_typed(st_convert(bytes, off, n, dtype))
}

///|
/// Matrix weight upload: BF16 payload uploaded verbatim (no conversion —
/// kernels unpack bf16 to the exact same f32 values as the CPU oracle).
fn upload_w(
  g : @gpu.Gpu,
  st : @qwen3.SafeTensors,
  name : String,
) -> @gpu.GPUBuffer raise Error {
  let (bytes, off, n, dtype) = st.raw(name)
  guard dtype == "BF16" else { raise Failure("\{name}: expected BF16") }
  g.upload_raw(bytes, off, n * 2)
}

///|
fn load_model(g : @gpu.Gpu, st : @qwen3.SafeTensors) -> Model raise Error {
  let w : Map[String, @gpu.GPUBuffer] = Map([])
  w["embed"] = upload_w(g, st, "model.embed_tokens.weight")
  w["norm"] = upload(g, st, "model.norm.weight")
  let kc : Array[@gpu.GPUBuffer] = []
  let vc : Array[@gpu.GPUBuffer] = []
  for i in 0..<NL {
    let p = "model.layers.\{i}"
    w["q\{i}"] = upload_w(g, st, "\{p}.self_attn.q_proj.weight")
    w["k\{i}"] = upload_w(g, st, "\{p}.self_attn.k_proj.weight")
    w["v\{i}"] = upload_w(g, st, "\{p}.self_attn.v_proj.weight")
    w["o\{i}"] = upload_w(g, st, "\{p}.self_attn.o_proj.weight")
    w["qn\{i}"] = upload(g, st, "\{p}.self_attn.q_norm.weight")
    w["kn\{i}"] = upload(g, st, "\{p}.self_attn.k_norm.weight")
    w["in\{i}"] = upload(g, st, "\{p}.input_layernorm.weight")
    w["pn\{i}"] = upload(g, st, "\{p}.post_attention_layernorm.weight")
    w["gate\{i}"] = upload_w(g, st, "\{p}.mlp.gate_proj.weight")
    w["up\{i}"] = upload_w(g, st, "\{p}.mlp.up_proj.weight")
    w["down\{i}"] = upload_w(g, st, "\{p}.mlp.down_proj.weight")
    kc.push(g.alloc(MAXPOS * NKH * HD))
    vc.push(g.alloc(MAXPOS * NKH * HD))
  }
  { w, kc, vc }
}

///|
/// RoPE cos/sin tables [MAXPOS, 64], f64 math matching the CPU oracle:
/// freq = theta^(-2j/hd) * pos.
fn rope_tables(g : @gpu.Gpu) -> (@gpu.GPUBuffer, @gpu.GPUBuffer) {
  let half = HD / 2
  let cost : FixedArray[Float] = FixedArray::make(MAXPOS * half, (0.0 : Float))
  let sint : FixedArray[Float] = FixedArray::make(MAXPOS * half, (0.0 : Float))
  for pos in 0..<MAXPOS {
    for j in 0..<half {
      let freq = @math.pow(1000000.0, -2.0 * j.to_double() / HD.to_double()) *
        pos.to_double()
      cost[pos * half + j] = Float::from_double(@math.cos(freq))
      sint[pos * half + j] = Float::from_double(@math.sin(freq))
    }
  }
  (g.upload(cost), g.upload(sint))
}

///|
/// Activation scratch, sized for the bucketed maximum (MAXPOS rows).
struct GPUAct {
  x : @gpu.GPUBuffer
  xn : @gpu.GPUBuffer
  qq : @gpu.GPUBuffer
  kk : @gpu.GPUBuffer
  vv : @gpu.GPUBuffer
  attn : @gpu.GPUBuffer
  proj : @gpu.GPUBuffer
  gb : @gpu.GPUBuffer
  ub : @gpu.GPUBuffer
  mlpb : @gpu.GPUBuffer
  /// Flash-decoding partial scratch: (m, l, acc[hd]) per (head, split).
  apart : @gpu.GPUBuffer
}

///|
fn alloc_act(g : @gpu.Gpu, rows : Int) -> GPUAct {
  {
    x: g.alloc(rows * HID),
    xn: g.alloc(rows * HID),
    qq: g.alloc(rows * NQH * HD),
    kk: g.alloc(rows * NKH * HD),
    vv: g.alloc(rows * NKH * HD),
    attn: g.alloc(rows * NQH * HD),
    apart: g.alloc(@gpu.Gpu::attn_decode_part_len(NQH, HD)),
    proj: g.alloc(rows * HID),
    gb: g.alloc(rows * INTER),
    ub: g.alloc(rows * INTER),
    mlpb: g.alloc(rows * HID),
  }
}

///|
/// Prefill layer (GEMM path; decode uses decode_layer). `rows` is the
/// bucketed row count (next pow2 >= prompt len) so shape-baked pipeline
/// keys (gemm/rope/embed/add/silu) don't recompile per prompt length.
/// zp is a permanently-zero u32 params buffer: prefill attention/rope
/// always start at cache position 0 (never the decode pos buffer).
fn layer(
  g : @gpu.Gpu,
  m : Model,
  li : Int,
  x : GPUAct,
  rows : Int,
  zp : @gpu.GPUBuffer,
  cost : @gpu.GPUBuffer,
  sint : @gpu.GPUBuffer,
) -> Unit {
  let s = rows
  let nq = NQH * HD
  let nkv = NKH * HD
  g.rmsnorm(x.x, m.w["in\{li}"], x.xn, s, HID, EPS)
  g.matmul(x.xn, m.w["q\{li}"], x.qq, s, nq, HID)
  g.matmul(x.xn, m.w["k\{li}"], x.kk, s, nkv, HID)
  g.matmul(x.xn, m.w["v\{li}"], x.vv, s, nkv, HID)
  // qk norm + rope WITHOUT in-place rmsnorm (aliasing = Dawn validation
  // error). Scratch reuse: normed q → attn (free until attention), normed
  // k → mlpb (free until the MLP; row width HID == nkv).
  g.rmsnorm(x.qq, m.w["qn\{li}"], x.attn, s * NQH, HD, EPS)
  g.rmsnorm(x.kk, m.w["kn\{li}"], x.mlpb, s * NKH, HD, EPS)
  g.rope(x.attn, cost, sint, zp, s, NQH, HD / 2)
  g.rope(x.mlpb, cost, sint, zp, s, NKH, HD / 2)
  g.copy(x.mlpb, 0, m.kc[li], 0, s * nkv)
  g.copy(x.vv, 0, m.vc[li], 0, s * nkv)
  // attention reads q from attn, writes its output into qq (now free)
  g.attn_prefill(x.attn, m.kc[li], m.vc[li], x.qq, zp, s, NQH, NKH, HD)
  g.matmul(x.qq, m.w["o\{li}"], x.proj, s, HID, nq)
  g.add(x.x, x.proj, s * HID)
  g.rmsnorm(x.x, m.w["pn\{li}"], x.xn, s, HID, EPS)
  g.matmul(x.xn, m.w["gate\{li}"], x.gb, s, INTER, HID)
  g.matmul(x.xn, m.w["up\{li}"], x.ub, s, INTER, HID)
  // silu(g)*u → mlpb (free), down proj → ub (free), residual from there:
  // keeps every binding pair alias-free
  g.silu_mul(x.gb, x.ub, x.mlpb, s * INTER)
  g.matmul(x.mlpb, m.w["down\{li}"], x.ub, s, HID, INTER)
  g.add(x.x, x.ub, s * HID)
}

///|
/// Fused decode layer (rows=1): 9 dispatches instead of 19 —
/// qkv+v→cache, qknorm+rope+k→cache, attention, o, add+rmsnorm,
/// gate+up, silu, down, add+rmsnorm(next layer's input norm; the last
/// layer uses the final norm so lm_head reads xn directly). All bindings
/// are alias-free (single-binding in-place only).
fn decode_layer(
  g : @gpu.Gpu,
  m : Model,
  li : Int,
  x : GPUAct,
  pos0 : @gpu.GPUBuffer,
  cost : @gpu.GPUBuffer,
  sint : @gpu.GPUBuffer,
) -> Unit {
  let nq = NQH * HD
  let nkv = NKH * HD
  g.qkv_fused(
    x.xn,
    m.w["q\{li}"],
    m.w["k\{li}"],
    m.w["v\{li}"],
    x.qq,
    x.kk,
    m.vc[li],
    pos0,
    nq,
    nkv,
    HID,
  )
  g.qknorm_rope(
    x.qq,
    x.kk,
    m.w["qn\{li}"],
    m.w["kn\{li}"],
    m.kc[li],
    cost,
    sint,
    pos0,
    NQH,
    NKH,
    HD,
    EPS,
  )
  g.attn_decode(
    x.qq,
    m.kc[li],
    m.vc[li],
    x.attn,
    x.apart,
    pos0,
    NQH,
    NKH,
    HD,
    MAXPOS,
  )
  g.matvec(m.w["o\{li}"], x.attn, x.proj, HID, nq)
  g.add_rmsnorm(x.x, x.proj, m.w["pn\{li}"], x.xn, 1, HID, EPS)
  g.gemv2_fused(x.xn, m.w["gate\{li}"], m.w["up\{li}"], x.gb, x.ub, INTER, HID)
  g.matvec_silu(m.w["down\{li}"], x.gb, x.ub, x.mlpb, HID, INTER)
  let next_norm = if li + 1 < NL { m.w["in\{li + 1}"] } else { m.w["norm"] }
  g.add_rmsnorm(x.x, x.mlpb, next_norm, x.xn, 1, HID, EPS)
}

///|
/// Next power of two >= s (min 8) — pipeline-key bucketing for prefill.
fn bucket(s : Int) -> Int {
  let mut b = 8
  while b < s {
    b = b * 2
  }
  b
}

///|
pub fn le_f32(b : Bytes, off : Int) -> Float {
  Float::reinterpret_from_int(
    b[off].to_int() |
    (b[off + 1].to_int() << 8) |
    (b[off + 2].to_int() << 16) |
    (b[off + 3].to_int() << 24),
  )
}

///|
pub fn argmax(logits : FixedArray[Float]) -> Int {
  let mut best : Float = -3.4e38
  let mut idx = 0
  for i in 0..<logits.length() {
    if logits[i] > best {
      best = logits[i]
      idx = i
    }
  }
  idx
}

///|
/// Qwen3 chat template (non-thinking), specials spliced as raw ids.
fn template_ids(tk : @qwen3.Tokenizer, q : String) -> Array[Int] raise Error {
  let out : Array[Int] = [151644] // <|im_start|>
  out.append(tk.encode("user\n" + q))
  out.push(151645) // <|im_end|>
  out.append(tk.encode("\n"))
  out.push(151644)
  out.append(tk.encode("assistant\n"))
  out
}

///|
/// Loaded model + tokenizer + all scratch state, ready for prefill/decode.
pub struct Runner {
  g : @gpu.Gpu
  model : Model
  tk : @qwen3.Tokenizer
  log : (String) -> Unit
  act : GPUAct
  xl : @gpu.GPUBuffer
  zp : @gpu.GPUBuffer
  pos0b : @gpu.GPUBuffer
  cost : @gpu.GPUBuffer
  sint : @gpu.GPUBuffer
  tok_buf : @gpu.GPUBuffer
  logits : @gpu.GPUBuffer
  am_v : @gpu.GPUBuffer
  am_i : @gpu.GPUBuffer
  am_o : @gpu.GPUBuffer
}

///|
/// Load weights + tokenizer from `dir` (via the host's async `read`) and
/// upload everything to the GPU. `log` reports progress/errors; on failure
/// the callback is NOT invoked (an error is logged instead).
pub fn Runner::boot(
  g : @gpu.Gpu,
  read : (String, (Bytes) -> Unit) -> Unit,
  dir : String,
  log : (String) -> Unit,
  cb : (Runner) -> Unit,
) -> Unit {
  read(dir + "/model.safetensors", fn(stb) {
    read(dir + "/tokenizer.json", fn(tjb) {
      let r = boot_from_bytes(g, stb, tjb, log) catch {
        e => {
          log("boot error: \{e}")
          return
        }
      }
      cb(r)
    })
  })
}

///|
fn boot_from_bytes(
  g : @gpu.Gpu,
  stb : Bytes,
  tjb : Bytes,
  log : (String) -> Unit,
) -> Runner raise Error {
  let t0 = now_ms()
  let st = @qwen3.SafeTensors::parse(stb)
  let model = load_model(g, st)
  let tk = @qwen3.Tokenizer::load(bytes_to_string(tjb))
  let (cost, sint) = rope_tables(g)
  let pos0b = g.alloc_u32()
  g.set_u32(pos0b, 0)
  let zp = g.alloc_u32()
  g.set_u32(zp, 0)
  g.sync(fn(_) { () })
  log("weights uploaded: \{(now_ms() - t0).to_int()} ms")
  {
    g,
    model,
    tk,
    log,
    act: alloc_act(g, MAXPOS),
    xl: g.alloc(HID),
    zp,
    pos0b,
    cost,
    sint,
    tok_buf: g.alloc_u32(),
    logits: g.alloc(VOCAB),
    am_v: g.alloc(594),
    am_i: g.alloc(594),
    am_o: g.alloc(1),
  }
}

///|
/// Plain BPE encode (no chat template) — for the correctness gate.
pub fn Runner::encode(self : Runner, text : String) -> Array[Int] raise Error {
  self.tk.encode(text)
}

///|
/// Batched prefill: embeds + NL layers + final norm + lm_head → logits
/// buffer (single encoder, single submit). Rows are bucketed to the next
/// pow2 to keep shape-baked pipeline keys hot.
pub fn Runner::prefill(self : Runner, ids : Array[Int]) -> Unit {
  let g = self.g
  let s = ids.length()
  let sb = bucket(s)
  let ids_buf = g.upload_u32(ids)
  g.begin()
  g.embed_rows(self.model.w["embed"], ids_buf, self.act.x, sb, HID)
  for li in 0..<NL {
    layer(g, self.model, li, self.act, sb, self.zp, self.cost, self.sint)
  }
  g.copy(self.act.x, (s - 1) * HID, self.xl, 0, HID)
  // final norm: out to act.xn (alias-free), lm_head reads from there
  g.rmsnorm(self.xl, self.model.w["norm"], self.act.xn, 1, HID, EPS)
  g.matvec(self.model.w["embed"], self.act.xn, self.logits, VOCAB, HID)
  g.flush()
}

///|
/// Read back the VOCAB logits produced by the last prefill.
pub fn Runner::readback_logits(
  self : Runner,
  cb : (FixedArray[Float]) -> Unit,
) -> Unit {
  self.g.readback(self.logits, VOCAB, cb)
}

///|
/// Sampling hyperparameters. `temp <= 0` = greedy (GPU argmax fast path).
pub struct Sampler {
  temp : Double
  top_k : Int
  top_p : Double
}

///|
pub fn Sampler::greedy() -> Sampler {
  { temp: 0.0, top_k: 1, top_p: 1.0 }
}

///|
pub fn Sampler::new(temp : Double, top_k : Int, top_p : Double) -> Sampler {
  { temp, top_k, top_p }
}

///|
pub fn Sampler::temp(self : Sampler) -> Double {
  self.temp
}

///|
pub fn Sampler::top_k(self : Sampler) -> Int {
  self.top_k
}

///|
pub fn Sampler::top_p(self : Sampler) -> Double {
  self.top_p
}

///|
/// CPU sampling over the full logits: temperature scale → top-k → nucleus
/// (top-p) → softmax draw. Only used when `temp > 0`; the greedy path
/// stays on the GPU argmax (no 607KB logits readback per token).
fn sample_logits(
  logits : FixedArray[Float],
  temp : Double,
  top_k : Int,
  top_p : Double,
) -> Int {
  let n = logits.length()
  // (scaled logit, id) pairs, descending by logit
  let pairs : Array[(Double, Int)] = Array::makei(n, fn(i) {
    (logits[i].to_double() / temp, i)
  })
  pairs.sort_by(fn(a, b) { b.0.compare(a.0) })
  let keep = if top_k > 0 && top_k < n { top_k } else { n }
  let mx = pairs[0].0
  let mut sum = 0.0
  let probs : Array[Double] = []
  let ids : Array[Int] = []
  for i in 0..<keep {
    let p = @math.exp(pairs[i].0 - mx)
    probs.push(p)
    ids.push(pairs[i].1)
    sum += p
  }
  // nucleus: cut the tail beyond top_p of the cumulative mass
  let mut cutoff = probs.length()
  let mut acc = 0.0
  for i in 0..<probs.length() {
    acc += probs[i] / sum
    if acc >= top_p {
      cutoff = i + 1
      break
    }
  }
  let mut cum = 0.0
  for i in 0..<cutoff {
    cum += probs[i]
  }
  let mut r = randf() * cum
  for i in 0..<cutoff {
    r -= probs[i]
    if r <= 0.0 {
      return ids[i]
    }
  }
  ids[cutoff - 1]
}

///|
/// Greedy decode one token, then recurse from the argmax callback.
fn decode_step(
  r : Runner,
  cur : Int,
  pos : Int,
  remaining : Int,
  out : Array[Int],
  sd : @qwen3.StreamDecoder,
  on_token : (String) -> Unit,
  sampler : Sampler,
  t0 : Double,
  verbose : Bool,
  done : (String, Int) -> Unit,
) -> Unit {
  if remaining == 0 || cur == 151643 || cur == 151645 {
    let text = r.tk.decode(out) catch { _ => "<decode error>" }
    done(text, out.length())
    return
  }
  let g = r.g
  let ts = now_ms()
  g.set_u32(r.tok_buf, cur)
  g.set_u32(r.pos0b, pos)
  g.begin()
  g.embed_rows(r.model.w["embed"], r.tok_buf, r.act.x, 1, HID)
  g.rmsnorm(r.act.x, r.model.w["in0"], r.act.xn, 1, HID, EPS)
  for li in 0..<NL {
    decode_layer(g, r.model, li, r.act, r.pos0b, r.cost, r.sint)
  }
  // act.xn already carries the final norm (last layer's add_rmsnorm)
  g.matvec(r.model.w["embed"], r.act.xn, r.logits, VOCAB, HID)
  g.flush()
  let te = now_ms()
  let advance = fn(am : Int) {
    if verbose {
      (r.log)(
        "step pos=\{pos}: encode=\{(te - ts).to_int()}ms total=\{(now_ms() - ts).to_int()}ms",
      )
    }
    out.push(am)
    let piece = sd.push(am) catch { _ => "" }
    if piece != "" {
      on_token(piece)
    }
    decode_step(
      r,
      am,
      pos + 1,
      remaining - 1,
      out,
      sd,
      on_token,
      sampler,
      t0,
      verbose,
      done,
    )
  }
  if sampler.temp <= 0.0 {
    g.argmax(r.logits, VOCAB, r.am_v, r.am_i, r.am_o, advance)
  } else {
    g.readback(r.logits, VOCAB, fn(got) {
      advance(sample_logits(got, sampler.temp, sampler.top_k, sampler.top_p))
    })
  }
}

///|
/// Decode `max` tokens starting from `first` at position `pos` (stops
/// early at <|endoftext|>/<|im_end|>). `sampler.temp <= 0` = greedy GPU
/// argmax; otherwise CPU temperature/top-k/top-p sampling. on_token fires
/// with UTF-8-safe incremental text; on_done(text, n_tokens, ms).
pub fn Runner::decode_greedy(
  self : Runner,
  first : Int,
  pos : Int,
  max : Int,
  sampler : Sampler,
  verbose : Bool,
  on_token : (String) -> Unit,
  on_done : (String, Int, Int) -> Unit,
) -> Unit {
  let t0 = now_ms()
  let sd = @qwen3.StreamDecoder::new(self.tk)
  decode_step(self, first, pos, max, [], sd, on_token, sampler, t0, verbose, fn(
    text,
    n,
  ) {
    on_done(text, n, (now_ms() - t0).to_int())
  })
}

///|
/// One chat turn: chat-template `user`, prefill, greedy-decode up to `max`
/// tokens. on_done(text, prefill_ms, decode_ms, n_tokens); failures arrive
/// as a bracketed "[...]" text with zeroed stats.
pub fn Runner::ask(
  self : Runner,
  user : String,
  max : Int,
  sampler : Sampler,
  on_token : (String) -> Unit,
  on_done : (String, Int, Int, Int) -> Unit,
) -> Unit {
  let ids = template_ids(self.tk, user) catch {
    e => {
      on_done("[tokenize error: \{e}]", 0, 0, 0)
      return
    }
  }
  if ids.length() > MAXPOS - max - 8 {
    on_done("[prompt too long: \{ids.length()} tokens]", 0, 0, 0)
    return
  }
  let tp = now_ms()
  self.prefill(ids)
  self.readback_logits(fn(got) {
    let pms = (now_ms() - tp).to_int()
    let am = argmax(got)
    self.decode_greedy(am, ids.length(), max, sampler, false, on_token, fn(
      text,
      n,
      dms,
    ) {
      on_done(text, pms, dms, n)
    })
  })
}

///|
fn rand_vals(n : Int, seed : Int) -> FixedArray[Float] {
  FixedArray::makei(n, fn(i) {
    Float::from_int((i * 37 + seed * 91 + 13) % 29 - 14) / (14.0 : Float)
  })
}

///|
fn diff(a : FixedArray[Float], b : FixedArray[Float]) -> Double {
  let mut d : Double = 0.0
  for i in 0..<a.length() {
    let e = (a[i] - b[i]).to_double().abs()
    if e > d {
      d = e
    }
  }
  d
}

///|
/// Op-level checks for the kernels (CPU loops as oracle), run before the
/// model: isolates kernel bugs from model wiring bugs.
pub fn check_kernels(
  g : @gpu.Gpu,
  log : (String) -> Unit,
  next : (Unit) -> Unit,
) -> Unit {
  // rmsnorm rows=2 n=128
  let rn = 128
  let x = rand_vals(2 * rn, 1)
  let w = rand_vals(rn, 2)
  let xb = g.upload(x)
  let wb = g.upload(w)
  let yb = g.alloc(2 * rn)
  g.rmsnorm(xb, wb, yb, 2, rn, EPS)
  g.readback(yb, 2 * rn, fn(y) {
    let want : FixedArray[Float] = FixedArray::make(2 * rn, (0.0 : Float))
    for r in 0..<2 {
      let mut ss = 0.0
      for i in 0..<rn {
        let v = x[r * rn + i].to_double()
        ss += v * v
      }
      let inv = Float::from_double(1.0 / (ss / 128.0 + 1.0e-6).sqrt())
      for i in 0..<rn {
        want[r * rn + i] = x[r * rn + i] * inv * w[i]
      }
    }
    log("check rmsnorm: \{diff(y, want)}")
    // rope rows=1 heads=2 half=4 (hd=8), pos0=3
    let half = 4
    let hd8 = 8
    let rx = rand_vals(2 * hd8, 3)
    let rxb = g.upload(rx)
    let cost : FixedArray[Float] = FixedArray::make(16, (0.0 : Float))
    let sint : FixedArray[Float] = FixedArray::make(16, (0.0 : Float))
    for pos in 0..<4 {
      for j in 0..<half {
        let freq = @math.pow(1000000.0, -2.0 * j.to_double() / 8.0) *
          pos.to_double()
        cost[pos * half + j] = Float::from_double(@math.cos(freq))
        sint[pos * half + j] = Float::from_double(@math.sin(freq))
      }
    }
    let costb = g.upload(cost)
    let sintb = g.upload(sint)
    let pos0 = g.alloc_u32()
    g.set_u32(pos0, 3)
    g.rope(rxb, costb, sintb, pos0, 1, 2, half)
    g.readback(rxb, 2 * hd8, fn(got) {
      let want = rx.copy()
      for head in 0..<2 {
        for j in 0..<half {
          let c = cost[3 * half + j]
          let s = sint[3 * half + j]
          let a = rx[head * hd8 + j]
          let b = rx[head * hd8 + j + half]
          want[head * hd8 + j] = a * c - b * s
          want[head * hd8 + j + half] = a * s + b * c
        }
      }
      log("check rope: \{diff(got, want)}")
      // silu_mul + add (out-of-place: aliasing is a Dawn validation error)
      let gn = rand_vals(32, 4)
      let un = rand_vals(32, 5)
      let gb = g.upload(gn)
      let ub = g.upload(un)
      let sb = g.alloc(32)
      g.silu_mul(gb, ub, sb, 32)
      g.add(sb, ub, 32)
      g.readback(sb, 32, fn(got) {
        let want : FixedArray[Float] = FixedArray::make(32, (0.0 : Float))
        for i in 0..<32 {
          let gv = gn[i]
          let sil = gv /
            ((1.0 : Float) + Float::from_double(@math.exp(-gv.to_double())))
          want[i] = sil * un[i] + un[i]
        }
        log("check silu+add: \{diff(got, want)}")
        // flash attention: hq=2 hkv=1 hd=128 max_pos=8, shared oracle.
        // prefill: rows=2 pos0=0 (causal); decode: row 0, pos0=1 (m=2),
        // 16 splits over 2 keys exercises the empty-split path.
        let hq = 2
        let hd = 128
        let q = rand_vals(2 * hq * hd, 6)
        let kc = rand_vals(8 * hd, 7)
        let vc = rand_vals(8 * hd, 8)
        let qb = g.upload(q)
        let kb = g.upload(kc)
        let vb = g.upload(vc)
        let ob = g.alloc(2 * hq * hd)
        let zp = g.alloc_u32()
        g.set_u32(zp, 0)
        g.attn_prefill(qb, kb, vb, ob, zp, 2, hq, 1, hd)
        // oracle: causal softmax attention in f64
        let want : FixedArray[Float] = FixedArray::make(
          2 * hq * hd,
          (0.0 : Float),
        )
        let scale = 1.0 / 128.0.sqrt()
        for r in 0..<2 {
          for head in 0..<hq {
            let m = r + 1
            let mut mx = -3.0e38
            let sc : FixedArray[Double] = FixedArray::make(m, 0.0)
            for p in 0..<m {
              let mut acc = 0.0
              for d in 0..<hd {
                acc = acc +
                  q[(r * hq + head) * hd + d].to_double() *
                  kc[p * hd + d].to_double()
              }
              sc[p] = acc * scale
              if sc[p] > mx {
                mx = sc[p]
              }
            }
            let mut sum = 0.0
            for p in 0..<m {
              sc[p] = @math.exp(sc[p] - mx)
              sum = sum + sc[p]
            }
            for d in 0..<hd {
              let mut acc = 0.0
              for p in 0..<m {
                acc = acc + sc[p] * vc[p * hd + d].to_double()
              }
              want[(r * hq + head) * hd + d] = Float::from_double(acc / sum)
            }
          }
        }
        g.readback(ob, 2 * hq * hd, fn(got) {
          log("check attn_prefill: \{diff(got, want)}")
          // decode: q row 1 of the fixture (attends keys 0..1 at pos0=1)
          let qd : FixedArray[Float] = FixedArray::makei(hq * hd, fn(i) {
            q[hq * hd + i]
          })
          let qdb = g.upload(qd)
          let db = g.alloc(hq * hd)
          let pb = g.alloc(@gpu.Gpu::attn_decode_part_len(hq, hd))
          let p1 = g.alloc_u32()
          g.set_u32(p1, 1)
          g.attn_decode(qdb, kb, vb, db, pb, p1, hq, 1, hd, 8)
          g.readback(db, hq * hd, fn(got) {
            let wantd : FixedArray[Float] = FixedArray::makei(hq * hd, fn(i) {
              want[hq * hd + i]
            })
            log("check attn_decode: \{diff(got, wantd)}")
            next(())
          })
        })
      })
    })
  })
}
