///|
/// wasm SIMD (f32x4) kernels operating on little-endian f32 byte buffers.
/// Element index `i` maps to linear address `ptr + i * 4`.

///|
/// Dot product of two length-`n` f32 rows at element offsets `qi` / `ki`.
#borrow(q, k)
fn kdot(
  q : FixedArray[Byte],
  qi : Int,
  k : FixedArray[Byte],
  ki : Int,
  n : Int,
) -> Float =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (result f32) (local i32) (local v128) (local f32) f32.const 0 f32x4.splat local.set 6 i32.const 0 local.set 5 block loop local.get 5 local.get 4 i32.const 4 i32.sub i32.gt_s br_if 1 local.get 6 local.get 0 local.get 1 local.get 5 i32.add i32.const 2 i32.shl i32.add v128.load local.get 2 local.get 3 local.get 5 i32.add i32.const 2 i32.shl i32.add v128.load f32x4.mul f32x4.add local.set 6 local.get 5 i32.const 4 i32.add local.set 5 br 0 end end local.get 6 f32x4.extract_lane 0 local.get 6 f32x4.extract_lane 1 f32.add local.get 6 f32x4.extract_lane 2 local.get 6 f32x4.extract_lane 3 f32.add f32.add local.set 7 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 7 local.get 0 local.get 1 local.get 5 i32.add i32.const 2 i32.shl i32.add f32.load local.get 2 local.get 3 local.get 5 i32.add i32.const 2 i32.shl i32.add f32.load f32.mul f32.add local.set 7 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end local.get 7)

///|
/// acc[ab .. ab+n) += p * v[vb .. vb+n)
#borrow(acc, v)
fn kaxpy(
  acc : FixedArray[Byte],
  ab : Int,
  p : Float,
  v : FixedArray[Byte],
  vb : Int,
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param f32) (param i32) (param i32) (param i32) (local i32) (local v128) local.get 2 f32x4.splat local.set 7 i32.const 0 local.set 6 block loop local.get 6 local.get 5 i32.const 4 i32.sub i32.gt_s br_if 1 local.get 0 local.get 1 local.get 6 i32.add i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 6 i32.add i32.const 2 i32.shl i32.add v128.load local.get 7 local.get 3 local.get 4 local.get 6 i32.add i32.const 2 i32.shl i32.add v128.load f32x4.mul f32x4.add v128.store local.get 6 i32.const 4 i32.add local.set 6 br 0 end end block loop local.get 6 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 1 local.get 6 i32.add i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 6 i32.add i32.const 2 i32.shl i32.add f32.load local.get 2 local.get 3 local.get 4 local.get 6 i32.add i32.const 2 i32.shl i32.add f32.load f32.mul f32.add f32.store local.get 6 i32.const 1 i32.add local.set 6 br 0 end end)

///|
/// acc[ab .. ab+n) *= alpha
#borrow(acc)
fn kscale(acc : FixedArray[Byte], ab : Int, alpha : Float, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param f32) (param i32) (local i32) (local v128) local.get 2 f32x4.splat local.set 5 i32.const 0 local.set 4 block loop local.get 4 local.get 3 i32.const 4 i32.sub i32.gt_s br_if 1 local.get 0 local.get 1 local.get 4 i32.add i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 4 i32.add i32.const 2 i32.shl i32.add v128.load local.get 5 f32x4.mul v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 1 local.get 4 i32.add i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 4 i32.add i32.const 2 i32.shl i32.add f32.load local.get 2 f32.mul f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)
