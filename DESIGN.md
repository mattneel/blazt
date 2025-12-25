# blazt design validation (Phase 0)

This document is the **implementation plan and design record** for `blazt`, a linear-algebra library intended to eventually upstream as `std.linalg`.

For the full from-first-principles specification and background, see [`README.md`](README.md).

## Repository conventions

- **Library name**: `blazt` (future `std.linalg`)
- **Tests**: all tests live in `tests/` (keep `src/` implementation-only)
- **Quality gate**: must pass in both modes with **0 crashes** and **0 leaks**
  - Debug: `zig build test`
  - ReleaseFast: `zig build test -Doptimize=ReleaseFast`
- **Zig 0.16 gotchas**: [`docs/zig-gotchas.md`](docs/zig-gotchas.md)

## Goals (what we optimize for)

1. **Comptime specialization** (no runtime dispatch): SIMD width, tile sizes, and kernel variants selected at compile time from `@import("builtin").cpu`.
2. **Zero-cost abstractions**: generic code should compile to hand-written kernels.
3. **Allocator-aware**: all heap allocation is explicit; no hidden allocations in hot paths.
4. **Composable kernels**: fast path built from inlined micro-kernels + packing.
5. **Predictable performance**: bounded loops, explicit scratch buffers, no surprise slow paths.

## Non-goals (for now)

- Full upstream `std` integration (naming/stdlib constraints handled later).
- Perfect cross-arch parity on day 1 (we’ll start with x86_64 + NEON-friendly shapes, then tune).
- “Bitwise identical” floating-point parity for all ops (we use tolerance policies where association differs).

## Data model: Matrix layout + stride

`blazt.Matrix(T, layout)` represents a dense matrix with:

- `rows`, `cols`
- `stride` (leading dimension)
- `data: []align(CacheLine) T`

### Layout semantics

- **Row-major**: `A[i][j]` stored at `i * stride + j`, default `stride = cols`
- **Col-major**: `A[i][j]` stored at `j * stride + i`, default `stride = rows`

### Invariants

- `rows * cols == data.len` for owned matrices (views may differ later)
- `stride >= (layout == row_major ? cols : rows)`
- All `data` pointers for owned matrices are `CacheLine` aligned
- All indexing is bounds-checked via assertions in Debug

## Numeric policy (float mode + stability + tolerances)

### Float mode

- Default: keep Zig’s default float mode (strict) for correctness.
- Allowed: `@setFloatMode(.optimized)` **only inside tightly scoped kernels** (e.g. GEMM micro-kernel) where:
  - inputs/outputs are well-defined
  - NaN/Inf handling is documented
  - oracle parity tests use an appropriate tolerance

### Stability policy

- BLAS1 reductions (`dot`, `nrm2`, `asum`) must not regress badly vs naive reference on large vectors.
- Strategy:
  - SIMD accumulation in registers
  - tree reduction for vector lanes
  - optional compensation (Kahan / Neumaier) for final scalar accumulation where warranted

### Tolerance policy (oracle parity)

When comparing against OpenBLAS/BLIS:

- `f64`: default `rel = 1e-12`, `abs = 1e-12` (tighten/loosen per op)
- `f32`: default `rel = 1e-5`, `abs = 1e-6`
- For operations with different associativity (parallel reductions), use ULP- or relative-tolerance comparisons (not bitwise).

## GEMM design (packing + tiling + tails)

### Resource napkin math (sanity checks)

GEMM \(C_{M×N} = A_{M×K} B_{K×N}\) does \(2MNK\) FLOPs.

- **Arithmetic intensity** increases with matrix size and effective blocking.
- For large matrices, GEMM should be compute-bound if packing avoids cache/TLB thrash.

We target “within 2×” of a conservative theoretical peak for first cut; tuning comes later.

### Packing contracts

We will pack panels into `CacheLine` aligned contiguous buffers:

- `packA`: `MC×KC` panel packed into micro-panels of `MR×KC` (MR contiguous per k)
- `packB`: `KC×NC` panel packed into micro-panels of `KC×NR` (NR contiguous per k)

### Tail handling

We’ll support arbitrary `M/N/K` via:

- remainder micro-kernels for partial `MR`/`NR`, or
- padded packing (zero-fill) with guarded stores, or
- scalar fallback for very small remainder tiles

The exact strategy is tuned after correctness is established.

## Parallelism plan (thread pool + decomposition)

### Thread pool responsibilities

Thread pool is responsible for:

- reusable worker threads (no spawn-per-op)
- bounded queues / work stealing for load balancing
- `submit()` + `waitAll()` semantics

### GEMM decomposition

First parallel strategy (simple + low false-sharing risk):

- split across **N dimension** (column panels of `B` and corresponding columns of `C`)
- each worker operates on disjoint `C` regions
- per-thread scratch buffers aligned and padded to avoid false sharing

## Test strategy

- Unit tests for core types (`memory`, `matrix`) live under `tests/`
- Boundary tests: 0/1 sizes, tails, stride edge cases
- Parity tests: run when oracle libs are available; skip otherwise
- Benchmarks: tracked separately from unit tests, but still run in CI modes

## Open questions (tracked in beads)

- SVE: dynamic vector length strategy (`blazt-10o.5.14`)
- Exact tolerance policy per op and per float mode


