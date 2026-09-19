// WebGPU host for bench/gpu: runs the MoonBit JS bundle on a real GPU via
// Deno's wgpu (Vulkan) backend — no browser, works headless.
//
// Usage: moon build --target js && deno run --allow-read scripts/bench_gpu_host.js
//
// The bundle posts benchmark lines via fetch("/result"); print them instead.
globalThis.fetch = async (url, opt) => {
  if (String(url).startsWith("/result")) {
    console.log("RESULT:", opt?.body ?? "");
    return new Response("ok");
  }
  return fetch(url);
};

const bundle = new TextDecoder().decode(
  await Deno.readFile(new URL("../_build/js/debug/build/bench/gpu/gpu.js", import.meta.url)),
);
eval(bundle);
