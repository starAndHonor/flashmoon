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
        r.ask(text, 256, fn(answer, pms, dms, n) {
          busy.val = false
          ui_log(answer)
          ui_status(
            "prefill \{pms} ms · decode \{dms} ms · \{n} tok (\{if n > 0 { dms / n } else { 0 }} ms/tok)",
          )
        })
      })
    })
  }, log=ui_log)
}
