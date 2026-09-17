///|
/// Minimal WebGPU JS binding — only what the attention kernel needs
/// (JS target: browser). One-line extern wrappers over the WebGPU API.

///|
pub type GPU

///|
pub type GPUAdapter

///|
pub type GPUDevice

///|
pub type GPUBuffer

///|
pub type GPUShaderModule

///|
pub type GPUBindGroupLayout

///|
pub type GPUBindGroup

///|
pub type GPUPipelineLayout

///|
pub type GPUComputePipeline

///|
pub type GPUCommandEncoder

///|
pub type GPUComputePassEncoder

///|
pub type GPUCommandBuffer

///|
pub type GPUQueue

///|
pub extern "js" fn gpu() -> GPU =
  #|() => navigator.gpu

///|
extern "js" fn GPU::request_adapter_cb(
  self : GPU,
  cb : (GPUAdapter) -> Unit,
) -> Unit =
  #|(self, cb) => self.requestAdapter({ powerPreference: "high-performance" }).then(cb)

///|
pub fn GPU::request_adapter(self : GPU, cb : (GPUAdapter) -> Unit) -> Unit {
  self.request_adapter_cb(cb)
}

///|
/// Request a device with the adapter's maximum buffer limits (lm_head f32
/// needs a 608MB storage binding, far above the 128MB default).
extern "js" fn GPUAdapter::request_device_cb(
  self : GPUAdapter,
  cb : (GPUDevice) -> Unit,
) -> Unit =
  #|(self, cb) => {
  #|  const l = self.limits;
  #|  self.requestDevice({ requiredLimits: {
  #|    maxBufferSize: l.maxBufferSize,
  #|    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
  #|    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
  #|    maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
  #|  } }).then(cb);
  #|}

///|
pub fn GPUAdapter::request_device(
  self : GPUAdapter,
  cb : (GPUDevice) -> Unit,
) -> Unit {
  self.request_device_cb(cb)
}

///|
pub extern "js" fn GPUDevice::create_buffer(
  self : GPUDevice,
  size : Int,
  usage : Int,
  mapped_at_creation? : Bool = false,
) -> GPUBuffer =
  #|(self, size, usage, mappedAtCreation) => self.createBuffer({ size, usage, mappedAtCreation })

///|
pub extern "js" fn GPUBuffer::set_float_array(
  self : GPUBuffer,
  source : FixedArray[Float],
) -> Unit =
  #|(self, source) => { new Float32Array(self.getMappedRange()).set(source); }

///|
pub extern "js" fn GPUBuffer::get_float_array(
  self : GPUBuffer,
) -> FixedArray[Float] =
  #|(self) => Array.from(new Float32Array(self.getMappedRange()))

///|
pub extern "js" fn GPUBuffer::unmap(self : GPUBuffer) -> Unit =
  #|(self) => self.unmap()

///|
pub extern "js" fn GPUDevice::max_workgroup_storage(self : GPUDevice) -> Int =
  #|(self) => self.limits.maxComputeWorkgroupStorageSize

///|
pub extern "js" fn GPUDevice::on_uncaptured_error(
  self : GPUDevice,
  cb : (String) -> Unit,
) -> Unit =
  #|(self, cb) => self.addEventListener("uncapturederror", (e) => cb(e.error.message))

///|
pub extern "js" fn GPUAdapter::info_string(self : GPUAdapter) -> String =
  #|(self) => self.info ? `${self.info.vendor}|${self.info.architecture}|${self.info.device || ""}|${self.info.description || ""}` : "noinfo"

///|
pub extern "js" fn GPUAdapter::limit_string(self : GPUAdapter) -> String =
  #|(self) => `maxBuffer=${self.limits.maxBufferSize} maxStorageBinding=${self.limits.maxStorageBufferBindingSize}`

///|
/// Host-side f32 tensor (JS Float32Array) — unlike MoonBit
/// FixedArray[Float] on js, typed arrays have no 2^27 element cap.
pub type F32Buf

///|
pub extern "js" fn F32Buf::byte_len(self : F32Buf) -> Int =
  #|(self) => self.byteLength

///|
/// queue.writeBuffer(buf, off, typedArray) — any size, no JS-array cap.
pub extern "js" fn GPUQueue::write_buffer_f32(
  self : GPUQueue,
  buf : GPUBuffer,
  off : Int,
  data : F32Buf,
) -> Unit =
  #|(self, buf, off, data) => self.writeBuffer(buf, off, data)

///|
/// Write one u32 (kernel params like rope/attention position base).
pub extern "js" fn GPUQueue::write_u32(
  self : GPUQueue,
  buf : GPUBuffer,
  v : Int,
) -> Unit =
  #|(self, buf, v) => self.writeBuffer(buf, 0, new Uint32Array([v]))

///|
extern "js" fn GPUBuffer::map_async_cb(
  self : GPUBuffer,
  mode : Int,
  cb : (Bool) -> Unit,
) -> Unit =
  #|(self, mode, cb) => self.mapAsync(mode).then(cb)

///|
pub fn GPUBuffer::map_async(
  self : GPUBuffer,
  mode : Int,
  cb : (Bool) -> Unit,
) -> Unit {
  self.map_async_cb(mode, cb)
}

///|
pub extern "js" fn GPUDevice::create_shader_module(
  self : GPUDevice,
  code : String,
) -> GPUShaderModule =
  #|(self, code) => self.createShaderModule({ code })

///|
/// Create a compute pipeline with layout 'auto' plus the matching bind
/// group: binding 0..2 = read-only inputs (q, k, v), 3 = output.
pub extern "js" fn GPUDevice::create_pipeline_and_bind_group(
  self : GPUDevice,
  module_ : GPUShaderModule,
  entry_point : String,
  b0 : GPUBuffer,
  b1 : GPUBuffer,
  b2 : GPUBuffer,
  b3 : GPUBuffer,
) -> GPUBindGroupWithPipeline =
  #|(self, module, entryPoint, b0, b1, b2, b3) => {
  #|  const pipeline = self.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
  #|  const bindGroup = self.createBindGroup({
  #|    layout: pipeline.getBindGroupLayout(0),
  #|    entries: [
  #|      { binding: 0, resource: { buffer: b0 } },
  #|      { binding: 1, resource: { buffer: b1 } },
  #|      { binding: 2, resource: { buffer: b2 } },
  #|      { binding: 3, resource: { buffer: b3 } },
  #|    ],
  #|  });
  #|  return { pipeline, bindGroup };
  #|}

///|
/// Same as `create_pipeline_and_bind_group` but for two buffers:
/// binding 0 = read-only partials, 1 = output.
pub extern "js" fn GPUDevice::create_pipeline_and_bind_group2(
  self : GPUDevice,
  module_ : GPUShaderModule,
  entry_point : String,
  b0 : GPUBuffer,
  b1 : GPUBuffer,
) -> GPUBindGroupWithPipeline =
  #|(self, module, entryPoint, b0, b1) => {
  #|  const pipeline = self.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
  #|  const bindGroup = self.createBindGroup({
  #|    layout: pipeline.getBindGroupLayout(0),
  #|    entries: [
  #|      { binding: 0, resource: { buffer: b0 } },
  #|      { binding: 1, resource: { buffer: b1 } },
  #|    ],
  #|  });
  #|  return { pipeline, bindGroup };
  #|}

///|
/// Bare compute pipeline (layout auto) — cacheable by shader+dims key.
pub extern "js" fn GPUDevice::create_pipeline(
  self : GPUDevice,
  module_ : GPUShaderModule,
  entry_point : String,
) -> GPUComputePipeline =
  #|(self, module, entryPoint) => self.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })

///|
/// Bind group for a pipeline: binding i = bufs[i].
pub extern "js" fn GPUDevice::make_bind_group(
  self : GPUDevice,
  pipeline : GPUComputePipeline,
  bufs : FixedArray[GPUBuffer],
) -> GPUBindGroup =
  #|(self, pipeline, bufs) => self.createBindGroup({
  #|  layout: pipeline.getBindGroupLayout(0),
  #|  entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  #|})

///|
pub type GPUBindGroupWithPipeline

///|
pub extern "js" fn GPUBindGroupWithPipeline::pipeline(
  self : GPUBindGroupWithPipeline,
) -> GPUComputePipeline =
  #|(self) => self.pipeline

///|
pub extern "js" fn GPUBindGroupWithPipeline::bind_group(
  self : GPUBindGroupWithPipeline,
) -> GPUBindGroup =
  #|(self) => self.bindGroup

///|
pub extern "js" fn GPUDevice::create_command_encoder(
  self : GPUDevice,
) -> GPUCommandEncoder =
  #|(self) => self.createCommandEncoder()

///|
pub extern "js" fn GPUCommandEncoder::begin_compute_pass(
  self : GPUCommandEncoder,
) -> GPUComputePassEncoder =
  #|(self) => self.beginComputePass()

///|
pub extern "js" fn GPUComputePassEncoder::set_pipeline(
  self : GPUComputePassEncoder,
  pipeline : GPUComputePipeline,
) -> Unit =
  #|(self, pipeline) => self.setPipeline(pipeline)

///|
pub extern "js" fn GPUComputePassEncoder::set_bind_group(
  self : GPUComputePassEncoder,
  index : Int,
  bind_group : GPUBindGroup,
) -> Unit =
  #|(self, index, bindGroup) => self.setBindGroup(index, bindGroup)

///|
pub extern "js" fn GPUComputePassEncoder::dispatch_workgroups(
  self : GPUComputePassEncoder,
  x : Int,
  y : Int,
  z : Int,
) -> Unit =
  #|(self, x, y, z) => self.dispatchWorkgroups(x, y, z)

///|
pub extern "js" fn GPUComputePassEncoder::end(
  self : GPUComputePassEncoder,
) -> Unit =
  #|(self) => self.end()

///|
pub extern "js" fn GPUCommandEncoder::copy_buffer_to_buffer(
  self : GPUCommandEncoder,
  source : GPUBuffer,
  source_offset : Int,
  destination : GPUBuffer,
  destination_offset : Int,
  size : Int,
) -> Unit =
  #|(self, source, sourceOffset, destination, destinationOffset, size) => self.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size)

///|
pub extern "js" fn GPUCommandEncoder::finish(
  self : GPUCommandEncoder,
) -> GPUCommandBuffer =
  #|(self) => self.finish()

///|
pub extern "js" fn GPUDevice::get_queue(self : GPUDevice) -> GPUQueue =
  #|(self) => self.queue

///|
pub extern "js" fn GPUQueue::submit(
  self : GPUQueue,
  command_buffers : Array[GPUCommandBuffer],
) -> Unit =
  #|(self, commandBuffers) => self.submit(commandBuffers)

///|
/// Resolves (cb ignored-arg) when all submitted work has finished.
pub extern "js" fn GPUQueue::on_submitted_done(
  self : GPUQueue,
  cb : (Bool) -> Unit,
) -> Unit =
  #|(self, cb) => self.onSubmittedWorkDone().then(cb)

///|
/// Raw host bytes view (Uint8Array) — for zero-conversion weight upload
/// (BF16 payloads stored verbatim, unpacked in-kernel).
pub type RawBuf

///|
pub extern "js" fn RawBuf::byte_len(self : RawBuf) -> Int =
  #|(self) => self.byteLength

///|
/// Byte view over a Bytes payload (js Bytes = Uint8Array).
pub extern "js" fn bytes_view(
  data : Bytes,
  off : Int,
  byte_len : Int,
) -> RawBuf =
  #|(data, off, byte_len) => new Uint8Array(data.buffer, data.byteOffset + off, byte_len)

///|
/// f32 -> BF16 truncation (u16 array), for feeding test matrices to the
/// bf16 weight kernels.
pub extern "js" fn to_bf16(data : FixedArray[Float]) -> RawBuf =
  #|(data) => { const u = new Uint16Array(data.length); const f = new Float32Array(1); const b = new Uint32Array(f.buffer); for (let i = 0; i < data.length; i++) { f[0] = data[i]; u[i] = b[0] >>> 16; } return u; }

///|
pub extern "js" fn GPUQueue::write_buffer_raw(
  self : GPUQueue,
  buf : GPUBuffer,
  data : RawBuf,
) -> Unit =
  #|(self, buf, data) => self.writeBuffer(buf, 0, data)

///|
/// Upload an array of u32 (e.g. token ids for the embedding gather).
pub extern "js" fn GPUQueue::write_u32_array(
  self : GPUQueue,
  buf : GPUBuffer,
  data : Array[Int],
) -> Unit =
  #|(self, buf, data) => self.writeBuffer(buf, 0, new Uint32Array(data))
