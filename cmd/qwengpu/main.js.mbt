// SPDX-License-Identifier: Apache-2.0
// Copyright Text: 2026 starandhonor

///| Qwen3-0.6B WebGPU chat — Deno REPL frontend (js target).
///| Runner core lives in starAndHonor/flashmoon/qwenrun.

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
extern "js" fn print_raw(s : String) -> Unit =
  #| (s) => { Deno.stdout.writeSync(new TextEncoder().encode(s)); }


///|
extern "js" fn bytes_to_string(b : Bytes) -> String =
  #|(b) => new TextDecoder().decode(b)
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
/// Runner::boot expects an async read; Deno's is synchronous — shim it.
fn read_async(path : String, cb : (Bytes) -> Unit) -> Unit {
  cb(deno_read(path))
}

///|
/// Correctness gate: self-tokenize the HF reference prompt, prefill, and
/// compare logits against refs/qwen3-0.6b/ref_logits.f32.
fn gate(r : @qwenrun.Runner) -> Unit raise Error {
  let ids = r.encode("The capital of France is")
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
  let tp = now_ms()
  r.prefill(ids)
  r.readback_logits(fn(got) {
    post("prefill \{ids.length()} tokens: \{(now_ms() - tp).to_int()} ms")
    let refb = deno_read("refs/qwen3-0.6b/ref_logits.f32")
    let mut max_diff : Double = 0.0
    for i in 0..<got.length() {
      let d = (got[i] - @qwenrun.le_f32(refb, i * 4)).to_double().abs()
      if d > max_diff {
        max_diff = d
      }
    }
    let am = @qwenrun.argmax(got)
    post("gpu max_diff=\{max_diff}")
    post("gpu argmax=\{am} (expect 12095 ' Paris')")
    if max_diff < 0.1 && am == 12095 {
      post("MATCH")
    } else {
      post("MISMATCH")
      return
    }
    // sample decode of the gate prompt, then drop into the REPL
    r.decode_greedy(am, ids.length(), 16, true, fn(text, n, ms) {
      post("[decode \{n} tok: \{ms} ms]")
      post("text: \"\{text}\"")
      repl(r)
    })
  })
}

///|
fn repl(r : @qwenrun.Runner) -> Unit {
  print_raw("\n>>> ")
  read_line(fn(line) {
    if line == "quit" || line == "exit" {
      post("bye")
      return
    }
    if line.trim() == "" {
      repl(r)
      return
    }
    r.ask(line, 256, fn(text, pms, dms, n) {
      post("[\{pms}ms prefill / \{dms}ms decode, \{n} tok]")
      post("text: \"\{text}\"")
      repl(r)
    })
  }, fn() { post("bye") })
}

///|
fn main {
  @gpu.Gpu::init(fn(g) {
    @qwenrun.check_kernels(g, post, fn(_) {
      @qwenrun.Runner::boot(g, read_async, "refs/qwen3-0.6b", post, fn(r) {
        gate(r) catch {
          e => post("error: \{e}")
        }
      })
    })
  }, log=post)
}
