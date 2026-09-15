///|
/// WebGPU flash-attention forward driver (JS target: browser).
///
/// Uploads q/k/v to GPU storage buffers, dispatches the online-softmax
/// compute shader (one thread per query row), and reads back the output.

///|
extern "js" fn now_ms() -> Double =
  #|() => performance.now()

///|
const USAGE_STORAGE : Int = 0x0080

///|
const USAGE_COPY_DST : Int = 0x0008

///|
const USAGE_COPY_SRC : Int = 0x0004

///|
const USAGE_MAP_READ : Int = 0x0001

///|
const MAP_MODE_READ : Int = 0x0001

///|
fn extract2d(
  t : @nparray.NpArray,
  name : String,
) -> (FixedArray[Float], Int, Int) {
  let td = t.value()
  if td.dims.length() != 2 {
    abort("\{name}: expected 2D tensor")
  }
  (td.data, td.dims[0], td.dims[1])
}

///|
/// Create a storage buffer pre-filled with `data`.
fn upload(
  device : GPUDevice,
  data : FixedArray[Float],
  extra_usage : Int,
) -> GPUBuffer {
  let buf = device.create_buffer(
    data.length() * 4,
    USAGE_STORAGE | extra_usage,
    mapped_at_creation=true,
  )
  buf.set_float_array(data)
  buf.unmap()
  buf
}

///|
/// Flash attention forward on GPU: q [n,d], k [m,d], v [m,dv] → [n,dv].
/// Callback-style (MoonBit async intrinsics crash moonc v0.10.5 on js).
pub fn flash_attention_gpu(
  q : @nparray.NpArray,
  k : @nparray.NpArray,
  v : @nparray.NpArray,
  on_done : (@nparray.NpArray) -> Unit,
  log? : (String) -> Unit = fn(_) { () },
) -> Unit {
  let (qd, n, d) = extract2d(q, "q")
  let (kd, m, dk) = extract2d(k, "k")
  let (vd, vm, dv) = extract2d(v, "v")
  if d != dk || m != vm {
    abort("shape mismatch")
  }
  if dv > 128 {
    abort("dv > 128 not supported by the WGSL register array")
  }
  let t_start = now_ms()
  gpu().request_adapter(fn(adapter) {
    adapter.request_device(fn(device) {
      let t_device = now_ms()
      log(
        "device maxComputeWorkgroupStorageSize=\{device.max_workgroup_storage()}",
      )
      let smem_need = 16 * d * 4 + 16 * dv * 4 // BC=16 tiles for K and V
      device.on_uncaptured_error(fn(msg) { log("GPU error: \{msg}") })
      if smem_need > device.max_workgroup_storage() {
        abort(
          "shader needs \{smem_need}B workgroup storage, device allows \{device.max_workgroup_storage()}B",
        )
      }
      // Upload inputs; output buffer is storage + copy source
      let qb = upload(device, qd, 0)
      let kb = upload(device, kd, 0)
      let vb = upload(device, vd, 0)
      let outb = device.create_buffer(
        n * dv * 4,
        USAGE_STORAGE | USAGE_COPY_SRC,
        mapped_at_creation=false,
      )
      // Pipeline + bind group (layout auto)
      let shader = device.create_shader_module(attention_shader(n, m, d, dv))
      let pp = device.create_pipeline_and_bind_group(
        shader, "main", qb, kb, vb, outb,
      )
      let t_upload = now_ms()
      // Encode + submit
      let encoder = device.create_command_encoder()
      let pass = encoder.begin_compute_pass()
      pass.set_pipeline(pp.pipeline())
      pass.set_bind_group(0, pp.bind_group())
      pass.dispatch_workgroups((n + 63) / 64, 1, 1)
      pass.end()
      let readb = device.create_buffer(
        n * dv * 4,
        USAGE_MAP_READ | USAGE_COPY_DST,
        mapped_at_creation=false,
      )
      encoder.copy_buffer_to_buffer(outb, 0, readb, 0, n * dv * 4)
      device.get_queue().submit([encoder.finish()])
      let t_submit = now_ms()
      log(
        "  stages: init=\{t_device - t_start}ms upload+pipeline=\{t_upload - t_device}ms encode+submit=\{t_submit - t_upload}ms",
      )
      readb.map_async(MAP_MODE_READ, fn(_) {
        let t_done = now_ms()
        log("  gpu wait (kernel+copy+map): \{t_done - t_submit}ms")
        let result = readb.get_float_array()
        readb.unmap()
        on_done(@tensor.Tensor::from_host(result, [n, dv]))
      })
    })
  })
}
