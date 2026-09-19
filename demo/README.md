# demo/ — Qwen3-0.6B inference on flashmoon

Qwen3-0.6B 端到端推理,完全建立在 flashmoon 库(`gpu` + `flash`)之上——既是库的下游示例,也是其"真实可用性"的佐证:同一份核心代码零修改地运行在浏览器(交互式聊天页)、Deno(带数值自检的 REPL)、wasm 与 native 上。

## Components

| Package | Role |
|---|---|
| `demo/qwen` | safetensors 权重解析 + Qwen2 风格字节级 BPE 分词器(经 HuggingFace 参考输出逐条比对) |
| `demo/qwenrun` | 宿主无关的 runner 核心:bf16 权重加载、f32 KV cache、28 层 transformer 前向、贪心 + temperature/top-k/top-p 采样、流式输出;文件读取与日志通过注入适配 |

Executables live in `cmd/`: `qwencpu`(CPU 参考实现 + 分词器 oracle 测试)、`qwengpu`(Deno REPL)、`webchat`(浏览器页面)。

## Quick start

> Requires the model at `refs/Qwen3-0.6B/model.safetensors`
> (download from [HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)).

**💬 Browser chat**

```bash
moon build --target js
python3 -m http.server 8123
# Chrome / Edge with WebGPU →
open http://127.0.0.1:8123/cmd/webchat/chat.html
```

Drag the sliders (max / temp / top-k / top-p), press Send. `temp = 0` means greedy.

**🖥️ Deno REPL**

```bash
moon build --target js
deno run --allow-read scripts/qwengpu_host.js
```

| Command | Effect |
|---|---|
| `/max N` `/temp F` `/topk K` `/topp F` | set generation hyperparameters |
| `/greedy` | back to pure-GPU argmax |
| `quit` | exit |

**🧊 CPU reference runner**

```bash
moon run cmd/qwencpu --target native -- refs/Qwen3-0.6B/model.safetensors <ref_input.json> <ref_logits.f32>
```

## Performance

Measured on an RTX 4060 Laptop (release build, Qwen3-0.6B, short prompts):

| Surface | Prefill | Decode | Notes |
|---|---|---|---|
| **Chromium (WebGPU)** | ~300–410 ms | **~11 ms/tok (~90 tok/s)** greedy | ~36 ms/tok sampling (logits readback) |
| **Deno (WebGPU)** | ~350 ms | ~20 ms/tok greedy | second WGSL→Vulkan adapter layer |

Decode-critical invariants: f32-resident KV cache, RoPE fused on write, in-place
SiLU+residual, single-submit argmax — **only logits-per-token and final text
cross the GPU↔CPU boundary**.

## Verification

| Check | Result |
|---|---|
| GPU greedy-decode first token vs CPU reference (MATCH gate, every REPL startup) | **byte-exact** |
| Tokenizer vs HuggingFace oracle | exact ID match |
| Kernel unit checks (rmsnorm / rope / silu+add / attn_prefill / attn_decode, f64 oracle) | max diff ≤ 1.2e-7 |

The Deno REPL's MATCH gate compares the first token ID against the CPU
reference on every startup — kernel refactors that break numerics fail loudly.

## Known limitations

- Qwen3-0.6B only (architecture-general; weights/config hardcoded).
- Context capped at 2048 positions (KV-cache sizing constant `MAXPOS`; the flash kernels impose no score-scratch limit, so raising it is a one-line change plus re-test).
- No continuous batching / speculative decode; single sequence.
- Sampling reads the full logits back to CPU each token (~2.7× decode cost).
