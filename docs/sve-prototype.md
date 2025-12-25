# ARM SVE prototype notes (Zig 0.16)

## What SVE changes

ARM SVE is **vector-length agnostic**: the hardware vector length (VL) is chosen at runtime per CPU (typically 128–2048 bits). Correct, fast SVE code usually:

- uses **predication** for tails (no scalar remainder loops)
- uses **VL-agnostic loops** that scale with the runtime VL

## Zig 0.16 constraint

Zig `@Vector(N, T)` is **fixed-length at comptime**.

That means we cannot directly express LLVM scalable vectors (`<vscale x N x T>`) in normal Zig code today, so we cannot write a “true” VL-agnostic SVE micro-kernel purely with `@Vector` the way we can for fixed-width SIMD (x86/NEON).

Inline asm can emit SVE instructions, but it’s not a great surface for building the full kernel stack (predication, loads/stores, reductions, etc.) in a portable way, and it’s hard to test on non-SVE hosts.

## Practical strategy for blazt (prototype)

Use **runtime VL probing + dispatch** to a small set of fixed-width kernels:

1. At runtime (only when building for `aarch64+sve`), probe VL with `RDVL` to get VL in bytes.
2. Convert to element count for the operand type.
3. Choose the largest fixed-width kernel where `VL_fixed <= VL_runtime`.
4. Run that kernel in a loop; handle the tail with a scalar remainder (or later, with SVE predication).

This gives:

- **correctness** on any SVE VL (we never assume a larger VL than the CPU provides)
- **some** benefit from wider vectorization when VL > 128b
- a path forward to later “true” predicated kernels if Zig grows a scalable-vector surface

### Existing scaffolding in blazt

`blazt.sve` (see `src/sve.zig`) provides:

- `runtimeVlBytes()` / `runtimeVlElems(T)` (returns 0 if unavailable)
- `dispatchFixedVectorLen(candidates_desc, runtime_vl_elems, kernel, ctx)`

### Pseudocode: dispatch to fixed-width kernels

```zig
const blazt = @import("blazt");

fn kernel(comptime VL: usize, ctx: *Ctx) void {
    // comptime-fixed kernel instantiated for VL elements
}

pub fn run(ctx: *Ctx) void {
    const vl = blazt.sve.runtimeVlElems(f32); // 0 if not SVE
    _ = blazt.sve.dispatchFixedVectorLen(&.{ 16, 8, 4 }, vl, kernel, ctx);
}
```

## GEMM-specific notes

If we ever do SVE-aware GEMM:

- **MR** is the most natural dimension to tie to “vector lanes” for `f32`/`f64`
- **predicated tails** are preferred over scalar remainder micro-kernels
- **packing** stays the same conceptually, but the optimal `MR/NR` and block sizes likely differ for SVE (especially for larger VL)

For now, the library treats SVE-capable targets as “fixed-width SIMD” using NEON-shaped kernels/tuning unless the user explicitly opts into a future runtime-dispatch mode.


