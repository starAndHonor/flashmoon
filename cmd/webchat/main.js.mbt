// SPDX-License-Identifier: Apache-2.0
// Copyright Text: 2026 starandhonor

///| Qwen3-0.6B WebGPU chat — browser frontend (js target).
///| Serves chat.html; the runner core lives in qwenrun.

///|
extern "js" fn fetch_bytes(path : String, cb : (Bytes) -> Unit) -> Unit =
  #|(p, cb) => {
  #|  fetch(p).then(r => {
  #|    if (!r.ok) throw new Error(p + ': HTTP ' + r.status);
  #|    return r.arrayBuffer();
  #|  }).then(b => cb(new Uint8Array(b)))
  #|    .catch(e => ui_status('fetch failed: ' + e.message));
  #|}

///|
extern "js" fn ui_log(s : String) -> Unit =
  #|(s) => {
  #|  const d = document.createElement('div');
  #|  d.textContent = s;
  #|  const log = document.getElementById('log');
  #|  log.appendChild(d);
  #|  log.scrollTop = log.scrollHeight;
  #|}

///|
extern "js" fn ui_status(s : String) -> Unit =
  #|(s) => { document.getElementById('status').textContent = s; }

///|
extern "js" fn ui_answer_begin() -> Unit =
  #|() => {
  #|  const d = document.createElement('div');
  #|  d.className = 'answer';
  #|  document.getElementById('log').appendChild(d);
  #|  globalThis.__answer = d;
  #|}

///|
extern "js" fn ui_answer_append(s : String) -> Unit =
  #|(s) => {
  #|  if (!globalThis.__answer) return;
  #|  globalThis.__answer.textContent += s;
  #|  const log = document.getElementById('log');
  #|  log.scrollTop = log.scrollHeight;
  #|}

///|
extern "js" fn ui_get_cfg(cb : (Int, Double, Int, Double) -> Unit) -> Unit =
  #|(cb) => {
  #|  const v = (id, d) => { const x = parseFloat(document.getElementById(id).value); return isNaN(x) ? d : x; };
  #|  cb(Math.max(1, v('cfg_max', 256) | 0), v('cfg_temp', 0.0), Math.max(0, v('cfg_topk', 1) | 0), v('cfg_topp', 1.0));
  #|}

///|
extern "js" fn ui_enable() -> Unit =
  #|() => {
  #|  document.getElementById('in').disabled = false;
  #|  document.getElementById('send').disabled = false;
  #|  document.getElementById('in').focus();
  #|}

///|
extern "js" fn ui_on_send(cb : (String) -> Unit) -> Unit =
  #|(cb) => {
  #|  const inp = document.getElementById('in');
  #|  const go = () => {
  #|    const v = inp.value.trim();
  #|    if (!v) return;
  #|    inp.value = '';
  #|    cb(v);
  #|  };
  #|  document.getElementById('send').onclick = go;
  #|  inp.onkeydown = (e) => { if (e.key === 'Enter') go(); };
  #|}

///|
fn main {
  @gpu.Gpu::init(fn(g) {
    ui_status("downloading model (1.5 GB)…")
    @qwenrun.Runner::boot(g, fetch_bytes, "/refs/qwen3-0.6b", ui_log, fn(r) {
      ui_status("ready — Qwen3-0.6B on WebGPU")
      ui_enable()
      let busy : Ref[Bool] = { val: false }
      ui_on_send(fn(text) {
        if busy.val {
          return
        }
        busy.val = true
        ui_log(">>> " + text)
        ui_status("generating…")
        ui_answer_begin()
        ui_get_cfg(fn(max, temp, top_k, top_p) {
          let sampler = @qwenrun.Sampler::new(temp, top_k, top_p)
          r.ask(text, max, sampler, ui_answer_append, fn(_, pms, dms, n) {
            busy.val = false
            let mode = if temp <= 0.0 { "greedy" } else { "temp=\{temp} top_k=\{top_k} top_p=\{top_p}" }
            ui_status(
              "prefill \{pms} ms · decode \{dms} ms · \{n} tok (\{if n > 0 { dms / n } else { 0 }} ms/tok) · \{mode}",
            )
          })
        })
      })
    })
  }, log=ui_log)
}
