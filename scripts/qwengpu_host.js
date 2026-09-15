// WebGPU host for cmd/qwengpu: runs the MoonBit JS bundle on a real GPU via
// Deno's wgpu (Vulkan) backend — no browser, works headless.
//
// Usage: moon build --target js && deno run --allow-read scripts/qwengpu_host.js
globalThis.fetch = async (url, opt) => {
  if (String(url).startsWith("/result")) {
    console.log("RESULT:", opt?.body ?? "");
    return new Response("ok");
  }
  return fetch(url);
};

const bundle = new TextDecoder().decode(
  await Deno.readFile(
    new URL("../_build/js/debug/build/cmd/qwengpu/qwengpu.js", import.meta.url),
  ),
);
eval(bundle);
