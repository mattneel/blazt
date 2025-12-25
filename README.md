# `blazt` Specification

**Zig Version:** 0.16.0-dev  
**Status:** Proposal  
**Authors:** Community  
**Last Updated:** 2025-01

Development notes:
- Zig 0.16 API/syntax gotchas encountered in this repo: [`docs/zig-gotchas.md`](docs/zig-gotchas.md)
- Optimized Zig “northern star”: [`docs/kangarootwelve.zig`](docs/kangarootwelve.zig) — one of the most beautiful examples of optimized Zig we’ve ever seen; we should aspire to implement and refactor our ops/kernels/etc using the same idioms.
- Building with **native CPU features** (AVX2/AVX512/FMA, etc): `zig build test -Dcpu=native` (you can also pass a specific model like `-Dcpu=znver2`, and add/subtract features with `+feat/-feat` syntax).
- Inspecting build-time CPU cache constants: `zig build cpu-cache` then open `zig-out/cpu_cache.zig`.

---

## Abstract

A from-first-principles linear algebra library for Zig, designed to be the fastest possible implementation for the specific CPU compiling the code. No runtime dispatch, no lowest-common-denominator codegen. Every kernel is specialized at comptime for the target's SIMD width, cache hierarchy, and microarchitecture.

**Naming note:** This repository implements the design under the package name **`blazt`**. Longer-term, the goal is to upstream the design as `std.linalg`.

---

## Design Philosophy

1. **Comptime specialization over runtime polymorphism** — The target CPU is known at compile time. Use it.
2. **Zero-cost abstractions** — Generic code must compile to the same assembly as hand-written kernels.
3. **Allocator-aware** — All heap allocation is explicit and controllable.
4. **Composable primitives** — Build complex operations from fused, inlined micro-kernels.
5. **Predictable performance** — No hidden allocations, no surprising slow paths.

---

## Part I: Comptime Superpowers

### 1.1 Target Introspection at Comptime

Zig exposes the full target CPU at compile time via `@import("builtin")`:

```zig
const builtin = @import("builtin");
const cpu = builtin.cpu;

// Feature detection
const has_avx512f = cpu.features.isEnabled(.avx512f);
const has_avx2 = cpu.features.isEnabled(.avx2);
const has_fma = cpu.features.isEnabled(.fma);
const has_neon = cpu.features.isEnabled(.neon);
const has_sve = cpu.features.isEnabled(.sve);

// Cache hierarchy (when available)
const cache = cpu.cache;
const l1d_size = cache.l1d.size orelse 32 * 1024;
const l1d_line = cache.l1d.line_size orelse 64;
const l2_size = cache.l2.size orelse 256 * 1024;
const l3_size = cache.l3.size orelse 8 * 1024 * 1024;
```

**Usage:** All tiling factors, unroll depths, and SIMD widths are computed at comptime from these values. No runtime branches.

### 1.2 Comptime Tile Size Computation

The optimal tile size for GEMM is a function of:

- Register file size (determines micro-kernel dimensions)
- L1D cache size (determines micro-panel width)
- L2 cache size (determines macro-panel height)
- L3 cache size (determines block size for parallelism)

```zig
fn computeTileParams(comptime T: type) TileParams {
    const simd_width = std.simd.suggestVectorLength(T) orelse 4;
    const bytes_per_vec = simd_width * @sizeOf(T);

    // Micro-kernel: MR x NR block that fits in registers
    // Heuristic: ~80% of vector registers for C tile, rest for A/B streaming
    const num_vec_regs = comptime getVectorRegisterCount();
    const regs_for_c = num_vec_regs * 8 / 10;

    // Square-ish micro-tile, but favor NR for column-major C
    const mr = simd_width * 2;  // Rows of C (unrolled, scalar)
    const nr = simd_width * 4;  // Cols of C (vectorized)

    // KC: depth of micro-panel, sized to keep A/B panels in L1
    // A panel: MR x KC, B panel: KC x NR
    const l1_usable = l1d_size * 7 / 10; // Leave headroom
    const kc = @min(512, l1_usable / (2 * @sizeOf(T) * (mr + nr)));

    // MC: height of macro-panel A, sized for L2
    const l2_usable = l2_size * 7 / 10;
    const mc = l2_usable / (@sizeOf(T) * kc);

    // NC: width of macro-panel B, sized for L3 (or just large)
    const nc = l3_size / (@sizeOf(T) * kc);

    return .{ .mr = mr, .nr = nr, .kc = kc, .mc = mc, .nc = nc };
}
```

This computation happens **entirely at comptime**. The final binary contains only the optimal constants.

### 1.3 Comptime Loop Unrolling

Zig's `inline for` and `comptime` blocks allow explicit unrolling decisions:

```zig
fn microKernel(comptime MR: usize, comptime NR: usize, ...) void {
    // C accumulator tile lives in registers
    var c: [MR][NR / simd_width]@Vector(simd_width, T) = undefined;

    // Initialize from memory or zero
    inline for (0..MR) |i| {
        inline for (0..NR / simd_width) |j| {
            c[i][j] = if (beta == 0) @splat(0) else loadAligned(...);
        }
    }

    // Rank-1 updates, fully unrolled over MR/NR
    for (0..K) |k| {
        const a_col: [MR]T = loadAPanel(k);
        inline for (0..NR / simd_width) |j| {
            const b_vec = loadBPanel(k, j);
            inline for (0..MR) |i| {
                c[i][j] = @mulAdd(@as(@Vector(...), @splat(a_col[i])), b_vec, c[i][j]);
            }
        }
    }
}
```

The `inline for` loops are guaranteed to be unrolled. No loop overhead, optimal instruction scheduling.

### 1.4 Comptime-Generated Specialized Kernels

For critical operations, generate multiple specialized versions:

```zig
const KernelVariant = enum {
    generic,
    alpha_one,           // α = 1
    beta_zero,           // β = 0 (no read of C)
    alpha_one_beta_zero, // C = A·B
    alpha_one_beta_one,  // C += A·B
};

fn selectGemmKernel(comptime T: type, alpha: T, beta: T) *const KernelFn {
    return comptime blk: {
        if (alpha == 1 and beta == 0) break :blk &gemmKernel(T, .alpha_one_beta_zero);
        if (alpha == 1 and beta == 1) break :blk &gemmKernel(T, .alpha_one_beta_one);
        if (alpha == 1) break :blk &gemmKernel(T, .alpha_one);
        if (beta == 0) break :blk &gemmKernel(T, .beta_zero);
        break :blk &gemmKernel(T, .generic);
    };
}
```

When called with comptime-known scalars, this resolves to a direct call. No runtime branch.

---

## Part II: SIMD Architecture

### 2.1 `@Vector` Fundamentals

Zig's `@Vector(N, T)` is the foundation. It maps directly to hardware SIMD:

| Target   | `suggestVectorLength(f32)` | Hardware      |
| -------- | -------------------------- | ------------- |
| AVX-512  | 16                         | ZMM registers |
| AVX/AVX2 | 8                          | YMM registers |
| SSE      | 4                          | XMM registers |
| NEON     | 4                          | Q registers   |
| SVE      | scalable                   | Z registers   |

### 2.2 Critical `std.simd` Operations

```zig
const simd = std.simd;

// Horizontal operations (use sparingly — they serialize)
const sum = simd.reduce(.Add, vec);
const max = simd.reduce(.Max, vec);

// Shuffles (comptime shuffle mask)
const reversed = simd.shuffle(vec, undefined, .{7, 6, 5, 4, 3, 2, 1, 0});

// Interleave/deinterleave for AoS ↔ SoA
const lo = simd.interleaveLower(a, b);
const hi = simd.interleaveUpper(a, b);

// Prefetch (explicit cache control)
simd.prefetch(@ptrCast(ptr), .{
    .rw = .read,
    .locality = 3,  // L1
    .cache = .data,
});
```

### 2.3 FMA: The Inner Loop MVP

Fused multiply-add is non-negotiable for performance:

```zig
// WRONG: Two operations, rounding between them
result = a * b + c;

// RIGHT: Single FMA instruction, one rounding
result = @mulAdd(a, b, c);
```

For `@Vector` types, `@mulAdd` emits vector FMA (`vfmadd231ps`, `fmla`, etc.).

**Performance note:** On Haswell+, FMA has 0.5 CPI throughput. Two FMA units = 2 FMAs/cycle = 32 FLOPs/cycle (AVX2 f32).

### 2.4 Float Mode Control

```zig
// Per-function fast-math (use in inner kernels only)
fn hotKernel(...) callconv(.C) void {
    @setFloatMode(.optimized);  // Enables:
                                 // - No NaN/Inf checks
                                 // - Reassociation
                                 // - Reciprocal approximations
    // ... kernel body
}
```

**Warning:** Only enable in well-understood kernels. Document precision implications.

### 2.5 Scalable Vector Extension (SVE) Support

For ARM SVE, vector length is runtime-variable but comptime-queryable:

```zig
const sve_width = std.Target.arm.sve_vector_bits orelse 128;
const vl = sve_width / (@sizeOf(T) * 8);

// Use std.simd.Vector with dynamic count
fn sveKernel(comptime T: type, n: usize, a: [*]const T, b: [*]T) void {
    var i: usize = 0;
    while (i < n) : (i += vl) {
        const mask = i + vl <= n;  // Predication
        // SVE predicated load/store via LLVM intrinsics
    }
}
```

---

## Part III: Memory Hierarchy Optimization

### 3.1 Cache Line Alignment

The L1 cache line is 64 bytes on all modern x86/ARM. Misalignment causes:

- Split-line loads (2x latency)
- False sharing between cores
- Reduced prefetcher effectiveness

```zig
const CacheLine = 64;

// Force alignment on types
const AlignedF32x16 = struct {
    data: @Vector(16, f32) align(CacheLine),
};

// Alignment-aware allocation
fn allocAligned(allocator: Allocator, comptime T: type, n: usize) ![]align(CacheLine) T {
    return try allocator.alignedAlloc(T, CacheLine, n);
}

// Runtime alignment assertion
fn ensureAligned(ptr: anytype) @TypeOf(ptr) {
    if (@intFromPtr(ptr) % CacheLine != 0) {
        @panic("pointer not cache-line aligned");
    }
    return @alignCast(ptr);
}
```

### 3.2 Packing for Contiguous Access

GEMM performance depends on sequential memory access. Standard matrices have stride, which causes TLB misses and cache thrashing. Solution: **pack** tiles into contiguous buffers.

```zig
/// Pack a panel of A (MC x KC) into contiguous micro-panels (MR x KC)
fn packA(comptime T: type, comptime MR: usize, comptime MC: usize, comptime KC: usize,
         a: [*]const T, lda: usize, packed: [*]align(CacheLine) T) void {

    var p: usize = 0;
    var i: usize = 0;
    while (i < MC) : (i += MR) {
        for (0..KC) |k| {
            // Gather MR elements from column k into contiguous memory
            inline for (0..MR) |ii| {
                packed[p + ii] = a[(i + ii) * lda + k];
            }
            p += MR;
        }
    }
}
```

Packed format enables:

- Sequential reads in micro-kernel
- Perfect prefetching
- No TLB misses within a tile

### 3.3 Prefetching Strategy

Hardware prefetchers are good but not perfect. For GEMM:

```zig
fn microKernelWithPrefetch(...) void {
    for (0..K) |k| {
        // Prefetch next micro-panel of B (NR elements, 2 cache lines ahead)
        if (k + 2 < K) {
            const prefetch_dist = 2;
            std.simd.prefetch(@ptrCast(b_packed + (k + prefetch_dist) * NR), .{
                .rw = .read,
                .locality = 3,  // Keep in L1
                .cache = .data,
            });
        }

        // ... FMA operations
    }
}
```

### 3.4 Non-Temporal Stores

For write-only operations (initialization, copies), bypass cache:

```zig
fn streamingStore(comptime T: type, comptime N: usize, dst: [*]T, vec: @Vector(N, T)) void {
    // MOVNTPS/MOVNTPD — write directly to memory, no cache pollution
    asm volatile ("vmovntps %[vec], (%[dst])"
        :
        : [dst] "r" (dst),
          [vec] "x" (vec),
        : "memory"
    );
}
```

Use cases:

- Matrix initialization
- Out-of-place transpose
- Large memcpy

---

## Part IV: Lock-Free Parallelism

### 4.1 Thread Pool Architecture

Reuse threads, don't spawn per-operation:

```zig
const ThreadPool = struct {
    threads: []std.Thread,
    work_queue: LockFreeQueue(WorkItem),
    active_count: std.atomic.Value(u32),
    shutdown: std.atomic.Value(bool),

    fn init(allocator: Allocator, num_threads: ?usize) !*ThreadPool {
        const n = num_threads orelse (std.Thread.getCpuCount() catch 4);
        // ...
    }
};
```

### 4.2 Atomic Operations

Zig provides direct access to atomic operations:

```zig
// Load/Store
const val = @atomicLoad(.acquire, &shared_var);
@atomicStore(.release, &shared_var, new_val);

// Read-Modify-Write
const old = @atomicRmw(.Add, &counter, 1, .acq_rel);

// Compare-and-Swap
const result = @cmpxchgWeak(
    &ptr,
    expected,
    desired,
    .acq_rel,  // Success ordering
    .acquire,  // Failure ordering
);
```

### 4.3 Lock-Free Work Stealing Queue

For dynamic load balancing:

```zig
const WorkStealingDeque = struct {
    buffer: []std.atomic.Value(*WorkItem),
    top: std.atomic.Value(i64),    // Owner pushes/pops here
    bottom: std.atomic.Value(i64), // Thieves steal from here

    /// Owner: push work item (single-producer)
    fn push(self: *@This(), item: *WorkItem) void {
        const b = self.bottom.load(.relaxed);
        const t = self.top.load(.acquire);

        if (b - t >= self.buffer.len) {
            self.grow(); // Resize if full
        }

        self.buffer[@intCast(b % self.buffer.len)].store(item, .relaxed);
        std.atomic.fence(.release);
        self.bottom.store(b + 1, .relaxed);
    }

    /// Owner: pop work item (single-producer)
    fn pop(self: *@This()) ?*WorkItem {
        var b = self.bottom.load(.relaxed) - 1;
        self.bottom.store(b, .relaxed);
        std.atomic.fence(.seq_cst);
        var t = self.top.load(.relaxed);

        if (t <= b) {
            const item = self.buffer[@intCast(b % self.buffer.len)].load(.relaxed);
            if (t == b) {
                // Last item — race with stealers
                if (@cmpxchgStrong(&self.top, t, t + 1, .seq_cst, .relaxed) != null) {
                    self.bottom.store(b + 1, .relaxed);
                    return null;
                }
                self.bottom.store(b + 1, .relaxed);
            }
            return item;
        } else {
            self.bottom.store(b + 1, .relaxed);
            return null;
        }
    }

    /// Thief: steal work item (multi-consumer)
    fn steal(self: *@This()) ?*WorkItem {
        var t = self.top.load(.acquire);
        std.atomic.fence(.seq_cst);
        var b = self.bottom.load(.acquire);

        if (t < b) {
            const item = self.buffer[@intCast(t % self.buffer.len)].load(.relaxed);
            if (@cmpxchgStrong(&self.top, t, t + 1, .seq_cst, .relaxed) != null) {
                return null; // Lost race
            }
            return item;
        }
        return null;
    }
};
```

### 4.4 Parallel GEMM Decomposition

Partition the iteration space for parallelism:

```zig
/// Parallel GEMM: C[M×N] = α·A[M×K]·B[K×N] + β·C
fn gemmParallel(
    comptime T: type,
    m: usize, n: usize, k: usize,
    alpha: T,
    a: [*]const T, lda: usize,
    b: [*]const T, ldb: usize,
    beta: T,
    c: [*]T, ldc: usize,
    pool: *ThreadPool,
) void {
    const params = computeTileParams(T);

    // L3 blocking for parallelism
    const nc = params.nc;
    const mc = params.mc;

    // Partition N dimension across threads (column panels)
    var j: usize = 0;
    while (j < n) : (j += nc) {
        const jb = @min(nc, n - j);

        // Each thread gets a column panel of B and corresponding C columns
        pool.submit(struct {
            fn work(args: anytype) void {
                gemmMacroKernel(T, m, args.jb, k, alpha, a, lda,
                    b + args.j, ldb, beta, c + args.j, ldc);
            }
        }.work, .{ .j = j, .jb = jb });
    }

    pool.waitAll();
}
```

### 4.5 Avoiding False Sharing

When threads write to adjacent memory:

```zig
/// Per-thread accumulator with padding to prevent false sharing
const ThreadLocal = struct {
    data: [BufferSize]f32,
    _padding: [CacheLine - (BufferSize * 4) % CacheLine]u8 = undefined,

    comptime {
        std.debug.assert(@sizeOf(@This()) % CacheLine == 0);
    }
};

var thread_locals: []align(CacheLine) ThreadLocal = allocator.alignedAlloc(...);
```

---

## Part V: Specialized Operations

### 5.1 BLAS Level 1 (Vector-Vector)

All Level 1 operations are memory-bandwidth-bound. Optimize for streaming.

| Operation | Description | Key Optimization                        |
| --------- | ----------- | --------------------------------------- | --- | --------------------------- |
| `axpy`    | y = αx + y  | Fused loop, non-temporal prefetch       |
| `dot`     | x·y         | Tree reduction, minimize horizontal ops |
| `nrm2`    | ‖x‖₂        | Compensated summation (Kahan)           |
| `scal`    | x = αx      | Non-temporal stores                     |
| `copy`    | y = x       | `memcpy` or NT stores                   |
| `swap`    | x ↔ y       | Blocked for cache                       |
| `asum`    | Σ           | xᵢ                                      |     | SIMD absolute + tree reduce |
| `iamax`   | argmax      | xᵢ                                      |     | SIMD compare + blend        |

### 5.2 BLAS Level 2 (Matrix-Vector)

Memory-bound for large matrices. Blocked for cache.

| Operation | Description  | Key Optimization                         |
| --------- | ------------ | ---------------------------------------- |
| `gemv`    | y = αAx + βy | Row-major: dot products; Col-major: axpy |
| `ger`     | A = αxyᵀ + A | Outer product, column blocked            |
| `trsv`    | Solve Tx = b | Block-recursive for L1                   |
| `symv`    | y = αAx + βy | Exploit symmetry: halve bandwidth        |

### 5.3 BLAS Level 3 (Matrix-Matrix)

Compute-bound. This is where GEMM lives.

| Operation | Description          | Key Optimization                   |
| --------- | -------------------- | ---------------------------------- |
| `gemm`    | C = αAB + βC         | Full tiling, packing, micro-kernel |
| `syrk`    | C = αAAᵀ + βC        | Only compute lower/upper triangle  |
| `trsm`    | Solve TX = B         | Block-recursive + GEMM             |
| `symm`    | C = αAB + βC (A sym) | Unpack symmetric on-the-fly        |

### 5.4 LAPACK Decompositions

| Operation          | Description | Dependencies                    |
| ------------------ | ----------- | ------------------------------- |
| LU                 | PA = LU     | GEMM, TRSM, pivoting            |
| Cholesky           | A = LLᵀ     | SYRK, TRSM                      |
| QR                 | A = QR      | Householder, GEMM               |
| SVD                | A = UΣVᵀ    | Bidiagonalization, QR iteration |
| Eigendecomposition | A = VΛV⁻¹   | Hessenberg, QR shifts           |

All decompositions are built on Level 3 BLAS. Get GEMM right, and decompositions are fast.

---

## Part VI: API Surface

### 6.1 Matrix Types

```zig
/// Dense matrix with configurable layout and storage
pub fn Matrix(comptime T: type, comptime layout: Layout) type {
    return struct {
        data: []align(CacheLine) T,
        rows: usize,
        cols: usize,
        stride: usize,  // Leading dimension
        allocator: ?Allocator,

        pub const Layout = enum { row_major, col_major };
        pub const Element = T;

        pub fn init(allocator: Allocator, rows: usize, cols: usize) !@This() { ... }
        pub fn deinit(self: *@This()) void { ... }
        pub fn at(self: @This(), i: usize, j: usize) T { ... }
        pub fn atPtr(self: @This(), i: usize, j: usize) *T { ... }
        pub fn slice(self: @This(), row_range: Range, col_range: Range) @This() { ... }
        pub fn transpose(self: @This()) Matrix(T, oppositeLayout(layout)) { ... }
    };
}
```

### 6.2 Operation Namespace

```zig
pub const ops = struct {
    // Level 1
    pub fn axpy(comptime T: type, n: usize, alpha: T, x: []const T, y: []T) void;
    pub fn dot(comptime T: type, x: []const T, y: []const T) T;
    pub fn nrm2(comptime T: type, x: []const T) T;

    // Level 2
    pub fn gemv(comptime T: type, trans: Trans, m: usize, n: usize,
                alpha: T, a: Matrix(T), x: []const T, beta: T, y: []T) void;

    // Level 3
    pub fn gemm(comptime T: type, trans_a: Trans, trans_b: Trans,
                alpha: T, a: Matrix(T), b: Matrix(T), beta: T, c: *Matrix(T)) void;

    // LAPACK
    pub fn lu(comptime T: type, a: *Matrix(T), ipiv: []i32) LuError!void;
    pub fn cholesky(comptime T: type, uplo: UpLo, a: *Matrix(T)) CholeskyError!void;
    pub fn qr(comptime T: type, a: *Matrix(T), tau: []T) void;
    pub fn svd(comptime T: type, a: *Matrix(T), s: []T, u: ?*Matrix(T), vt: ?*Matrix(T)) void;
    pub fn eig(comptime T: type, a: *Matrix(T), eigenvalues: []Complex(T)) void;
};
```

### 6.3 Parallel Variants

```zig
pub const parallel = struct {
    pub fn gemm(comptime T: type, trans_a: Trans, trans_b: Trans,
                alpha: T, a: Matrix(T), b: Matrix(T), beta: T, c: *Matrix(T),
                thread_pool: *ThreadPool) void;

    // ... other parallel operations
};
```

### 6.4 Builder Pattern for Configuration

```zig
const result = linalg.gemm(f32)
    .transA(.no_trans)
    .transB(.trans)
    .alpha(2.0)
    .beta(1.0)
    .parallel(thread_pool)
    .execute(a, b, &c);
```

---

## Part VII: Testing and Benchmarking

### Project test layout (repo convention)

In this repository, **all tests live under `tests/`** so `src/` stays implementation-only and doesn’t carry inline `test {}` blocks.
The examples in this section are intended to live in files like `tests/gemm_parity_test.zig` and be aggregated by `tests/root.zig`.

**Quality gate:** the full test suite must pass in both modes with **0 crashes** and **0 leaks**:

- Debug: `zig build test`
- ReleaseFast: `zig build test -Doptimize=ReleaseFast`

### 7.1 Oracle Integration

OpenBLAS and BLIS are included as submodules for comparison:

```
deps/
├── openblas/     # git submodule
└── blis/         # git submodule
```

Runtime loading via `std.DynLib`:

```zig
const Oracle = struct {
    lib: std.DynLib,
    sgemm: *const fn(...) void,
    dgemm: *const fn(...) void,
    // ...

    pub fn load(path: []const u8) !Oracle {
        var lib = try std.DynLib.open(path);
        return .{
            .lib = lib,
            .sgemm = lib.lookup(*const fn(...) void, "cblas_sgemm")
                     orelse return error.SymbolNotFound,
            // ...
        };
    }
};
```

#### Providing oracle libraries (runtime)

The oracle loader will try common system sonames (e.g. `libopenblas.so`, `libblis.so`) automatically. You can also **override explicitly**:

- `BLAZT_ORACLE_LIB`: explicit path/name to the library to load
- `BLAZT_ORACLE_OPENBLAS`: path/name for OpenBLAS
- `BLAZT_ORACLE_BLIS`: path/name for BLIS

Tests that depend on an oracle will **skip** if no suitable library can be loaded.

### 7.2 Parity Testing

Every operation must produce bitwise-identical results (for non-associative ops, within ULP tolerance):

```zig
// tests/gemm_parity_test.zig
test "gemm parity with OpenBLAS" {
    const openblas = try Oracle.load("libopenblas.so");
    defer openblas.unload();

    // Generate random matrices
    var a = try randomMatrix(f64, 1024, 1024, allocator);
    var b = try randomMatrix(f64, 1024, 1024, allocator);
    var c_ours = try Matrix(f64).init(allocator, 1024, 1024);
    var c_oracle = try Matrix(f64).init(allocator, 1024, 1024);

    // Run both
    linalg.gemm(f64, .no_trans, .no_trans, 1.0, a, b, 0.0, &c_ours);
    openblas.dgemm(.col_major, .no_trans, .no_trans, 1024, 1024, 1024,
                   1.0, a.data.ptr, 1024, b.data.ptr, 1024, 0.0,
                   c_oracle.data.ptr, 1024);

    // Compare within tolerance
    for (0..1024*1024) |i| {
        try testing.expectApproxEqRel(c_ours.data[i], c_oracle.data[i], 1e-10);
    }
}
```

### 7.3 Benchmark Framework

```zig
const Benchmark = struct {
    name: []const u8,
    samples: []f64,  // nanoseconds per op

    fn run(comptime op: fn () void, warmup: usize, iterations: usize) Benchmark {
        // Warmup
        for (0..warmup) |_| op();

        // Timed runs
        var samples: [iterations]f64 = undefined;
        for (0..iterations) |i| {
            const start = std.time.nanoTimestamp();
            op();
            const end = std.time.nanoTimestamp();
            samples[i] = @floatFromInt(end - start);
        }

        return .{ .name = name, .samples = &samples };
    }

    fn gflops(self: Benchmark, flop_count: usize) f64 {
        const median_ns = percentile(self.samples, 0.5);
        return @as(f64, @floatFromInt(flop_count)) / median_ns;
    }
};
```

### 7.4 Performance Metrics

For GEMM (M×N×K):

- **FLOPs:** 2·M·N·K (multiply-add pairs)
- **Arithmetic Intensity:** 2·M·N·K / (M·K + K·N + M·N) × sizeof(T)
- **Peak GFLOPS:** cores × frequency × SIMD_width × 2 (FMA) × FMA_units

Report:

1. Achieved GFLOPS
2. Percentage of theoretical peak
3. Comparison to OpenBLAS/BLIS

---

## Part VIII: Implementation Checklist

### Phase 1: Foundation

- [ ] Target introspection (`CpuInfo` comptime struct)
- [ ] Aligned allocation utilities
- [ ] SIMD vector abstraction layer
- [ ] Thread pool with work stealing
- [ ] Benchmark harness
- [ ] Oracle loading (OpenBLAS, BLIS)

### Phase 2: Level 1 BLAS

- [ ] `copy`, `scal`, `swap`
- [ ] `axpy`
- [ ] `dot`
- [ ] `nrm2`, `asum`
- [ ] `iamax`, `iamin`

### Phase 3: Level 2 BLAS

- [ ] `gemv` (row-major, col-major)
- [ ] `ger`, `geru`, `gerc`
- [ ] `trsv`, `trmv`
- [ ] `symv`, `hemv`

### Phase 4: GEMM (The Big One)

- [ ] Micro-kernel (MR × NR)
- [ ] Packing routines (A, B)
- [ ] Macro-kernel (loop nest)
- [ ] Parallel decomposition
- [ ] Tuning per architecture (Intel, AMD, ARM)

### Phase 5: Level 3 BLAS (GEMM-based)

- [ ] `symm`, `hemm`
- [ ] `syrk`, `herk`
- [ ] `syr2k`, `her2k`
- [ ] `trmm`, `trsm`

### Phase 6: LAPACK

- [ ] LU decomposition (blocked)
- [ ] Cholesky decomposition
- [ ] QR decomposition
- [ ] SVD (divide-and-conquer)
- [ ] Eigendecomposition

---

## Appendix A: Target-Specific Tuning

### A.1 Intel (Haswell+)

```zig
const intel_haswell = struct {
    const simd_width = 8;  // AVX2
    const fma_units = 2;
    const mr = 6;   // 6 rows of C
    const nr = 16;  // 2 AVX registers wide
    const vec_regs = 16;
    const l1d = 32 * 1024;
    const l2 = 256 * 1024;
    const l3_per_core = 2 * 1024 * 1024;
};
```

### A.2 AMD (Zen4+)

```zig
const amd_zen4 = struct {
    const simd_width = 16;  // AVX-512
    const fma_units = 2;
    const mr = 8;
    const nr = 32;  // 2 ZMM registers wide
    const vec_regs = 32;
    const l1d = 32 * 1024;
    const l2 = 1024 * 1024;  // Larger L2
    const l3_per_ccx = 32 * 1024 * 1024;
};
```

### A.3 Apple Silicon (M1+)

```zig
const apple_m1 = struct {
    const simd_width = 4;  // NEON (128-bit)
    const fma_units = 4;   // Many execution units
    const mr = 8;
    const nr = 12;
    const vec_regs = 32;
    const l1d = 128 * 1024;  // Large L1
    const l2 = 12 * 1024 * 1024;
    // Unified memory: different prefetch strategy
};
```

---

## Appendix B: Inline Assembly Escape Hatch

When LLVM doesn't generate optimal code:

```zig
fn avx512_fma_8x16(
    a: *const [8]f32,
    b: *const @Vector(16, f32),
    c: *[8]@Vector(16, f32),
) void {
    asm volatile (
        \\vbroadcastss (%[a]), %%zmm16
        \\vfmadd231ps (%[b]), %%zmm16, %%zmm0
        \\vbroadcastss 4(%[a]), %%zmm17
        \\vfmadd231ps (%[b]), %%zmm17, %%zmm1
        // ... continue for all 8 rows
        :
        : [a] "r" (a),
          [b] "r" (b),
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm16", "zmm17", "memory"
    );
}
```

**Use sparingly.** Prefer intrinsics. Document why assembly is required.

---

## Appendix C: Float Stability Considerations

### Compensated Summation (Kahan)

For `nrm2`, `asum`, `dot` on large vectors:

```zig
fn kahanSum(comptime T: type, values: []const T) T {
    var sum: T = 0;
    var c: T = 0;  // Compensation

    for (values) |v| {
        const y = v - c;
        const t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
```

### Blocked Summation

For SIMD, use tree reduction followed by Kahan for final accumulation:

```zig
fn simdDot(comptime T: type, x: []const T, y: []const T) T {
    const V = @Vector(simd_width, T);
    var acc: V = @splat(0);

    var i: usize = 0;
    while (i + simd_width <= x.len) : (i += simd_width) {
        const xv: V = x[i..][0..simd_width].*;
        const yv: V = y[i..][0..simd_width].*;
        acc = @mulAdd(xv, yv, acc);
    }

    // Tree reduction within vector
    var result = simd.reduce(.Add, acc);

    // Scalar tail
    while (i < x.len) : (i += 1) {
        result += x[i] * y[i];
    }

    return result;
}
```

---

## Appendix D: Memory Layout Reference

### Row-Major (C order)

```
A[i][j] @ address = base + i * lda + j
Contiguous along j (columns)
```

### Column-Major (Fortran order)

```
A[i][j] @ address = base + j * lda + i
Contiguous along i (rows)
```

### Packed Symmetric (Upper)

```
A[i][j] for i ≤ j stored at: j*(j+1)/2 + i
```

### Packed Banded

```
AB[kl+ku+1+i-j][j] = A[i][j]
```

---

## References

1. Goto, K., van de Geijn, R. "Anatomy of High-Performance Matrix Multiplication" (2008)
2. Van Zee, F.G., van de Geijn, R. "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (2015)
3. Intel Intrinsics Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
4. ARM NEON Programmer's Guide
5. Zig Language Reference: https://ziglang.org/documentation/master/

---

_This specification is a living document. Implementation experience will inform revisions._
