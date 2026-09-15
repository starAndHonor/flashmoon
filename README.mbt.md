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
