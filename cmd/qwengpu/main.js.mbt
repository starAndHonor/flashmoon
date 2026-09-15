///|
/// Qwen3-0.6B full-model WebGPU forward (js target, Deno host).
///
/// Prefill via tiled GEMM + multi-row causal attention, decode via GEMV +
/// single-row attention; all weights GPU-resident f32 (BF16 upcast on
/// upload, exactly like the CPU oracle). Verified against the HF reference
/// logits in refs/qwen3-0.6b.

///|
extern "js" fn post(s : String) -> Unit =
  #|(s) => { fetch('/result', { method: 'POST', body: s }); }

///|
extern "js" fn now_ms() -> Double =
  #|() => performance.now()

///|
extern "js" fn deno_read(path : String) -> Bytes =
  #|(p) => Deno.readFileSync(p)

///|
extern "js" fn bytes_to_string(b : Bytes) -> String =
  #|(b) => new TextDecoder().decode(b)

///|
extern "js" fn print_raw(s : String) -> Unit =
  #| (s) => { Deno.stdout.writeSync(new TextEncoder().encode(s)); }

///|
extern "js" fn read_line(cb : (String) -> Unit, eof : () -> Unit) -> Unit =
  #| (cb, eof) => {
  #|   if (!globalThis.__rl) {
  #|     globalThis.__rl = {
  #|       reader: Deno.stdin.readable.pipeThrough(new TextDecoderStream()).getReader(),
  #|       pending: "",
  #|     };
  #|   }
  #|   const st = globalThis.__rl;
  #|   (async () => {
  #|     for (;;) {
  #|       const nl = st.pending.indexOf("\n");
  #|       if (nl >= 0) {
  #|         const line = st.pending.slice(0, nl);
  #|         st.pending = st.pending.slice(nl + 1);
  #|         cb(line);
  #|         return;
  #|       }
  #|       const r = await st.reader.read();
  #|       if (r.done) {
  #|         const rest = st.pending;
  #|         st.pending = "";
  #|         if (rest) cb(rest); else eof();
  #|         return;
  #|       }
  #|       st.pending += r.value;
  #|     }
  #|   })();
  #| }

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
/// Activation scratch for a token batch (rows = s).
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
    proj: g.alloc(rows * HID),
    gb: g.alloc(rows * INTER),
    ub: g.alloc(rows * INTER),
    mlpb: g.alloc(rows * HID),
  }
}

///|
///| Prefill layer (GEMM path; decode uses decode_layer). `rows` is the
///| bucketed row count (next pow2 >= prompt len) so shape-baked pipeline
///| keys (gemm/rope/embed/add/silu) don't recompile per prompt length.
///| zp is a permanently-zero u32 params buffer: prefill attention/rope
///| always start at cache position 0 (never the decode pos buffer).
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
  g.rmsnorm(x.qq, m.w["qn\{li}"], x.qq, s * NQH, HD, EPS)
  g.rmsnorm(x.kk, m.w["kn\{li}"], x.kk, s * NKH, HD, EPS)
  g.rope(x.qq, cost, sint, zp, s, NQH, HD / 2)
  g.rope(x.kk, cost, sint, zp, s, NKH, HD / 2)
  g.copy(x.kk, 0, m.kc[li], 0, s * nkv)
  g.copy(x.vv, 0, m.vc[li], 0, s * nkv)
  g.attn_rows(
    x.qq,
    m.kc[li],
    m.vc[li],
    x.attn,
    zp,
    s,
    NQH,
    NKH,
    HD,
    MAXPOS,
  )
  g.matmul(x.attn, m.w["o\{li}"], x.proj, s, HID, nq)
  g.add(x.x, x.proj, s * HID)
  g.rmsnorm(x.x, m.w["pn\{li}"], x.xn, s, HID, EPS)
  g.matmul(x.xn, m.w["gate\{li}"], x.gb, s, INTER, HID)
  g.matmul(x.xn, m.w["up\{li}"], x.ub, s, INTER, HID)
  g.silu_mul(x.gb, x.ub, x.gb, s * INTER)
  g.matmul(x.gb, m.w["down\{li}"], x.mlpb, s, HID, INTER)
  g.add(x.x, x.mlpb, s * HID)
}

///| Next power of two >= s (min 8) — pipeline-key bucketing for prefill.
fn bucket(s : Int) -> Int {
  let mut b = 8
  while b < s {
    b = b * 2
  }
  b
}

///|
fn le_f32(b : Bytes, off : Int) -> Float {
  Float::reinterpret_from_int(
    b[off].to_int() |
    (b[off + 1].to_int() << 8) |
    (b[off + 2].to_int() << 16) |
    (b[off + 3].to_int() << 24),
  )
}

///|
fn argmax(logits : FixedArray[Float]) -> Int {
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
/// Op-level checks for the M2 kernels (CPU loops as oracle), run before
/// the model: isolates kernel bugs from model wiring bugs.
fn check_kernels(g : @gpu.Gpu, next : (Unit) -> Unit) -> Unit {
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
    post("check rmsnorm: \{diff(y, want)}")
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
      post("check rope: \{diff(got, want)}")
      // silu_mul + add
      let gn = rand_vals(32, 4)
      let un = rand_vals(32, 5)
      let gb = g.upload(gn)
      let ub = g.upload(un)
      g.silu_mul(gb, ub, gb, 32)
      g.add(gb, ub, 32)
      g.readback(gb, 32, fn(got) {
        let want : FixedArray[Float] = FixedArray::make(32, (0.0 : Float))
        for i in 0..<32 {
          let gv = gn[i]
          let sil = gv /
            ((1.0 : Float) + Float::from_double(@math.exp(-gv.to_double())))
          want[i] = sil * un[i] + un[i]
        }
        post("check silu+add: \{diff(got, want)}")
        // attn_rows: hq=2 hkv=1 hd=128 max_pos=8 rows=2 pos0=0 (causal)
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
        g.attn_rows(qb, kb, vb, ob, zp, 2, hq, 1, hd, 8)
        g.readback(ob, 2 * hq * hd, fn(got) {
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
          post("check attn_rows: \{diff(got, want)}")
          next(())
        })
      })
    })
  })
}

///|
fn run(g : @gpu.Gpu) -> Unit raise Error {
  let t0 = now_ms()
  let st = @qwen3.SafeTensors::parse(
    deno_read("refs/qwen3-0.6b/model.safetensors"),
  )
  let model = load_model(g, st)
  let (cost, sint) = rope_tables(g)
  let pos0b = g.alloc_u32()
  g.set_u32(pos0b, 0)
  let zp = g.alloc_u32()
  g.set_u32(zp, 0)
  g.sync(fn(_) { () })
  post("weights uploaded: \{(now_ms() - t0).to_int()} ms")
  // prompt: tokenize with our own BPE; assert it reproduces the HF ids
  let tk = @qwen3.Tokenizer::load(
    bytes_to_string(deno_read("refs/qwen3-0.6b/tokenizer.json")),
  )
  let prompt = "The capital of France is"
  let ids = tk.encode(prompt)
  let ref_ids_json = @json.parse(
    bytes_to_string(deno_read("refs/qwen3-0.6b/ref_input.json")),
  )
  guard ref_ids_json is Array(ref_ids) else {
    raise Failure("ref_input.json: expected array")
  }
  if ref_ids.length() != ids.length() {
    raise Failure("tokenizer diverges from ref_input.json (length)")
  }
  for i in 0..<ids.length() {
    guard ref_ids[i] is Number(rv, ..) else {
      raise Failure("ref_input.json: not number")
    }
    if rv.to_int() != ids[i] {
      raise Failure("tokenizer diverges from ref_input.json at \{i}")
    }
  }
  post("prompt: \{ids.length()} tokens (self-tokenized, matches HF ids)")
  let s = ids.length()
  let logits = g.alloc(VOCAB)
  let act = alloc_act(g, MAXPOS)
  let xl = g.alloc(HID)
  let tok_buf = g.alloc_u32()
  // argmax scratch: 594 workgroup partials + 1-float output
  let am_v = g.alloc(594)
  let am_i = g.alloc(594)
  let am_o = g.alloc(1)
  // correctness gate: prefill the HF reference prompt, compare logits
  let tp = now_ms()
  prefill(g, model, ids, act, xl, zp, cost, sint, logits)
  g.readback(logits, VOCAB, fn(got) {
    post("prefill \{s} tokens: \{(now_ms() - tp).to_int()} ms")
    let refb = deno_read("refs/qwen3-0.6b/ref_logits.f32")
    let mut max_diff : Double = 0.0
    for i in 0..<VOCAB {
      let d = (got[i] - le_f32(refb, i * 4)).to_double().abs()
      if d > max_diff {
        max_diff = d
      }
    }
    let am = argmax(got)
    post("gpu max_diff=\{max_diff}")
    post("gpu argmax=\{am} (expect 12095 ' Paris')")
    if max_diff < 0.1 && am == 12095 {
      post("MATCH")
    } else {
      post("MISMATCH")
      return
    }
    // sample decode of the gate prompt, then drop into the REPL
    decode_step(
      g, model, act, cost, sint, pos0b, tok_buf, logits, am_v, am_i, am_o, tk,
      am, s, 16, [], now_ms(), true, fn() {
        chat(g, model, tk, act, xl, zp, pos0b, cost, sint, tok_buf, logits, am_v, am_i, am_o)
      },
    )
  })
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
/// Batched prefill: embeds + NL layers + final norm + lm_head → logits.
fn prefill(
  g : @gpu.Gpu,
  model : Model,
  ids : Array[Int],
  act : GPUAct,
  xl : @gpu.GPUBuffer,
  zp : @gpu.GPUBuffer,
  cost : @gpu.GPUBuffer,
  sint : @gpu.GPUBuffer,
  logits : @gpu.GPUBuffer,
) -> Unit {
  let s = ids.length()
  let sb = bucket(s)
  let ids_buf = g.upload_u32(ids)
  g.begin()
  g.embed_rows(model.w["embed"], ids_buf, act.x, sb, HID)
  for li in 0..<NL {
    layer(g, model, li, act, sb, zp, cost, sint)
  }
  g.copy(act.x, (s - 1) * HID, xl, 0, HID)
  g.rmsnorm(xl, model.w["norm"], xl, 1, HID, EPS)
  g.matvec(model.w["embed"], xl, logits, VOCAB, HID)
  g.flush()
}

///|
/// Interactive REPL: read a line, prefill the chat-templated prompt, greedy
/// decode up to 256 tokens (EOS stops early), print, loop.
fn chat(
  g : @gpu.Gpu,
  model : Model,
  tk : @qwen3.Tokenizer,
  act : GPUAct,
  xl : @gpu.GPUBuffer,
  zp : @gpu.GPUBuffer,
  pos0b : @gpu.GPUBuffer,
  cost : @gpu.GPUBuffer,
  sint : @gpu.GPUBuffer,
  tok_buf : @gpu.GPUBuffer,
  logits : @gpu.GPUBuffer,
  am_v : @gpu.GPUBuffer,
  am_i : @gpu.GPUBuffer,
  am_o : @gpu.GPUBuffer,
) -> Unit {
  print_raw("\n>>> ")
  read_line(fn(line) {
    if line == "quit" || line == "exit" {
      post("bye")
      return
    }
    let again = fn() {
      chat(g, model, tk, act, xl, zp, pos0b, cost, sint, tok_buf, logits, am_v, am_i, am_o)
    }
    if line.trim() == "" {
      again()
      return
    }
    let ids = template_ids(tk, line) catch {
      e => {
        post("tokenize error: \{e}")
        again()
        return
      }
    }
    if ids.length() > MAXPOS - 300 {
      post("prompt too long (\{ids.length()} tokens)")
      again()
      return
    }
    let tp = now_ms()
    prefill(g, model, ids, act, xl, zp, cost, sint, logits)
    g.readback(logits, VOCAB, fn(got) {
      post("[prefill \{ids.length()} tok: \{(now_ms() - tp).to_int()} ms]")
      let am = argmax(got)
      decode_step(
        g, model, act, cost, sint, pos0b, tok_buf, logits, am_v, am_i, am_o, tk,
        am, ids.length(), 256, [], now_ms(), false, again,
      )
    })
  }, fn() { post("bye") })
}

///|
/// Greedy decode one token, then recurse from the readback callback.
fn decode_step(
  g : @gpu.Gpu,
  model : Model,
  act : GPUAct,
  cost : @gpu.GPUBuffer,
  sint : @gpu.GPUBuffer,
  pos0b : @gpu.GPUBuffer,
  tok_buf : @gpu.GPUBuffer,
  logits : @gpu.GPUBuffer,
  am_v : @gpu.GPUBuffer,
  am_i : @gpu.GPUBuffer,
  am_o : @gpu.GPUBuffer,
  tk : @qwen3.Tokenizer,
  cur : Int,
  pos : Int,
  remaining : Int,
  out : Array[Int],
  t0 : Double,
  verbose : Bool,
  done : () -> Unit,
) -> Unit {
  if remaining == 0 || (cur == 151643 || cur == 151645) {
    let dt = now_ms() - t0
    if out.length() > 0 {
      post(
        "[decode \{out.length()} tok: \{dt.to_int()} ms (\{(dt / out.length().to_double()).to_int()} ms/tok)]",
      )
    }
    let text = tk.decode(out) catch { _ => "<decode error>" }
    post("text: \"\{text}\"")
    done()
    return
  }
  let ts = now_ms()
  g.set_u32(tok_buf, cur)
  g.set_u32(pos0b, pos)
  g.begin()
  g.embed_rows(model.w["embed"], tok_buf, act.x, 1, HID)
  g.rmsnorm(act.x, model.w["in0"], act.xn, 1, HID, EPS)
  for li in 0..<NL {
    decode_layer(g, model, li, act, pos0b, cost, sint)
  }
  // act.xn already carries the final norm (last layer's add_rmsnorm)
  g.matvec(model.w["embed"], act.xn, logits, VOCAB, HID)
  g.flush()
  let te = now_ms()
  g.argmax(logits, VOCAB, am_v, am_i, am_o, fn(am) {
    if verbose {
      post("step pos=\{pos}: encode=\{(te - ts).to_int()}ms total=\{(now_ms() - ts).to_int()}ms")
    }
    out.push(am)
    decode_step(
      g, model, act, cost, sint, pos0b, tok_buf, logits, am_v, am_i, am_o, tk,
      am, pos + 1, remaining - 1, out, t0, verbose, done,
    )
  })
}

///|
fn main {
  @gpu.Gpu::init(fn(g) {
    check_kernels(g, fn(_) {
      run(g) catch {
        e => post("error: \{e}")
      }
    })
  }, log=post)
}

///|
/// Fused decode layer (rows=1): 9 dispatches instead of 19 —
/// qkv+v→cache, qknorm+rope+k→cache, attention, o, add+rmsnorm,
/// gate+up, silu, down, add+rmsnorm(next layer's input norm; the last
/// layer uses the final norm so lm_head reads xn directly).
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
  g.attn_rows(x.qq, m.kc[li], m.vc[li], x.attn, pos0, 1, NQH, NKH, HD, MAXPOS)
  g.matvec(m.w["o\{li}"], x.attn, x.proj, HID, nq)
  g.add_rmsnorm(x.x, x.proj, m.w["pn\{li}"], x.xn, 1, HID, EPS)
  g.gemv2_fused(x.xn, m.w["gate\{li}"], m.w["up\{li}"], x.gb, x.ub, INTER, HID)
  g.matvec_silu(m.w["down\{li}"], x.gb, x.ub, x.mlpb, HID, INTER)
  let next_norm = if li + 1 < NL { m.w["in\{li + 1}"] } else { m.w["norm"] }
  g.add_rmsnorm(x.x, x.mlpb, next_norm, x.xn, 1, HID, EPS)
}
