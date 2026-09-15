# starAndHonor/flashmoon

FlashAttention-2 forward in pure MoonBit: online-softmax blocked kernels for
wasm/native (f32x4 SIMD), plus a WebGPU split-K path for the js target.

## GPU (WebGPU, js target)

`gpu.flash_attention_gpu(q, k, v, on_done, log?)`: q [n,d], k [m,d], v [m,dv]
→ [n,dv]. Fast path (d, dv ≤ 64, multiples of 4): split-K — kernel 1 computes
per-(row-block, key-chunk) online-softmax partials with vec4 shared-memory
tiles, kernel 2 merges them. Fallback for dv ≤ 128: single tiled kernel.

Run the browser-free benchmark/demo (real GPU via Deno's wgpu/Vulkan):

```sh
moon build --target js
deno run --allow-read scripts/webgpu_host.js
```

Measured on RTX 4060 Laptop (4096×4096, d=dv=64): kernels+copy ≈ 13ms
(~330 GFLOP/s), max_diff ≈ 2.4e-7 vs the CPU naive kernel.

## Qwen3-0.6B chat (WebGPU)

Full-model inference (`qwenrun` package: tokenizer, prefill GEMM path, fused
decode kernels, bf16 weights GPU-resident), verified against HF reference
logits (max_diff ≈ 4e-5).

Deno REPL (headless, real GPU via wgpu/Vulkan):

```sh
moon build --target js
deno run --allow-read scripts/qwengpu_host.js
```

Browser page (needs WebGPU flags on Linux Chrome):

```sh
moon build --target js
python3 -m http.server 8123   # repo root
chromium --enable-unsafe-webgpu --enable-features=Vulkan,WebGPUService,WebGPU \
  --use-webgpu-adapter=vulkan \
  http://127.0.0.1:8123/cmd/webchat/chat.html
```

Measured on RTX 4060 Laptop: prefill ~300 ms (short prompts), decode
~11 ms/tok in Chrome/Dawn (~30 ms/tok under Deno/wgpu). Chrome rejects
writable-storage binding aliasing that wgpu tolerates — all dispatches are
alias-free (single-binding in-place or out-of-place).
