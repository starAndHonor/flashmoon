// Learn more about moon.mod configuration:
// https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html
//
// To add a dependency, run this command in your terminal:
//   moon add moonbitlang/x
//
// Or manually declare it in `import`, for example:
// import {
//   "moonbitlang/x@0.4.6",
// }

name = "starAndHonor/flashmoon"

version = "0.1.0"

readme = "README.md"

repository = "https://github.com/starAndHonor/flashmoon"

license = "Apache-2.0"

keywords = [ "webgpu", "attention", "flashattention", "gpu", "inference" ]

preferred_target = "wasm"

description = "WebGPU-oriented AI inference foundation library: WebGPU runtime + WGSL kernel library, 4D batched FlashAttention (MHA/GQA/MQA, causal, cross-attention) on wasm/native/WebGPU, with verified numerics."

import {
  "chnlkw/moonxi-net@0.1.1",
  "moonbitlang/x@0.5.5",
}
