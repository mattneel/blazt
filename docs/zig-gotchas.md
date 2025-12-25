# Zig 0.16 gotchas (blazt)

This repo targets **Zig 0.16.0-dev**. Whenever we hit a Zig 0.16 API/syntax change while implementing **`blazt`**, we:

1. Find the source-of-truth in Zig’s stdlib/compiler sources
2. Fix the code and get `zig build test` passing (Debug + `-Doptimize=ReleaseFast`)
3. Document the pattern here

## `std.mem.Allocator.alignedAlloc` alignment is `?std.mem.Alignment` (not bytes)

- **Symptom**
  - `error: expected type '?mem.Alignment', found 'usize'` when calling `allocator.alignedAlloc(T, 64, n)`
- **Source**
  - `std/mem/Allocator.zig` defines:
    - `pub fn alignedAlloc(self: Allocator, comptime T: type, comptime alignment: ?Alignment, n: usize) ...`
  - `std.mem.Alignment` is an enum stored as log2(power-of-two) (see `std/mem.zig`).
- **Fix / pattern**

```zig
const alignment = std.mem.Alignment.fromByteUnits(64);
const buf = try allocator.alignedAlloc(u8, alignment, n);
```

Use `null` for “natural alignment”:

```zig
const buf = try allocator.alignedAlloc(u8, null, n);
```

## No `@Type(...)` builtin: prefer explicit type expressions

- **Symptom**
  - `error: invalid builtin function: '@Type'`
- **Pattern**
  - If you were constructing types from `@typeInfo(...)` (e.g. to “upgrade” pointer alignment), don’t. Instead, compute the values you need and return an explicit type expression.
  - Example pattern (used in `src/memory.zig`): switch on pointer _size_ and return `*`, `[*]`, `[]`, or `[*c]` with `align(N)` and qualifiers.

## `try` is not allowed inside `defer { ... }`

- **Symptom**
  - `error: 'try' not allowed inside defer expression`
- **Fix / pattern**
  - `defer` blocks can’t propagate errors. Use `catch` + `@panic`/`unreachable`, or restructure so the fallible check happens before returning.
  - For tests, prefer `std.testing.allocator` (leak checking is handled by the test runner in Debug mode in this repo).

## No `builtin.cpu.cache` in Zig 0.16: use build-time `cpu_cache` (or defaults) + `std.atomic.cache_line`

- **Symptom**
  - `error: no field named 'cache' in struct 'Target.Cpu'` when accessing `@import("builtin").cpu.cache`
- **Source**
  - `std/Target.zig` defines `pub const Cpu = struct { arch, model, features }` — **no cache hierarchy fields**
- **Fix / pattern**
  - In `blazt`, cache sizes come from a build-time generated module imported as `@import("cpu_cache")`:
    - native targets: probe via `tools/cpu_probe.zig` (x86 CPUID) during `zig build`
    - otherwise: fall back to `src/cpu_cache_default.zig` (conservative defaults)
  - Prefer the wrapper `CpuInfo.native()` (see `src/cpu.zig`) so callers get both feature flags and cache sizes.
  - For cache-line size, use `std.atomic.cache_line` (the stdlib’s canonical cache-line constant).

To inspect what the build detected on your machine:

- `zig build cpu-cache` then open `zig-out/cpu_cache.zig`

## CPU feature detection uses `std.Target.Cpu.has(...)` (arch-family aware)

- **Symptom**
  - `cpu.features.isEnabled(.avx2)` style code no longer works (the enum literal doesn’t match the new `isEnabled(Index)` signature).
- **Source**
  - `std/Target.zig` provides:
    - `pub fn has(cpu: Cpu, comptime family: Arch.Family, feature: @field(Target, @tagName(family)).Feature) bool`
- **Fix / pattern**

```zig
const builtin = @import("builtin");
const cpu = builtin.cpu;

const has_avx2 = cpu.has(.x86, .avx2);
const has_avx512f = cpu.has(.x86, .avx512f);
const has_fma = cpu.has(.x86, .fma);
const has_neon = cpu.has(.arm, .neon) or cpu.has(.aarch64, .neon);
const has_sve = cpu.has(.aarch64, .sve);
```

This returns `false` automatically when `cpu.arch.family()` doesn’t match the requested family.

## `@mulAdd` takes 4 arguments (type + 3 operands)

- **Symptom**
  - `error: expected 4 arguments, found 3` when calling `@mulAdd(a, b, c)`
- **Source**
  - `std/zig/BuiltinFn.zig` declares `@mulAdd` with `param_count = 4`
  - `std/math.zig` uses it as `@mulAdd(Type, a, b, c)`
- **Fix / pattern**

```zig
const r = @mulAdd(f64, x, y, z);
// or in a helper that infers type:
fn mulAdd(a: anytype, b: anytype, c: anytype) @TypeOf(a) {
    return @mulAdd(@TypeOf(a), a, b, c);
}
```

## `@shuffle` mask indices: second vector uses bitwise-not (`~i`) indices

- **Symptom**
  - `mask element ... selects out-of-bounds index` when using “concatenated indexing” like `N + i`
- **Source**
  - `std/simd.zig` uses `~iota(...)` to select elements from the second vector.
- **Fix / pattern**

```zig
// Select a[i] -> use i
// Select b[i] -> use ~i (bitwise not), which is negative for small i.
const a: @Vector(4, u32) = .{ 0, 1, 2, 3 };
const b: @Vector(4, u32) = .{ 10, 11, 12, 13 };
const out = @shuffle(u32, a, b, [_]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1) }); // {0,10,1,11}
```

## No `std.atomic.fence(...)` helper: use `seq_cst` atomic ops when you need a barrier

- **Symptom**
  - `error: root source file struct 'atomic' has no member named 'fence'`
- **Fix / pattern**
  - Zig 0.16’s `std.atomic` module does not expose `fence()`. When you need a full ordering point in a lock-free algorithm, use `seq_cst` operations on the relevant atomics (or restructure so the ordering is carried by acquire/release pairs).
  - Example (Chase–Lev pop path): use `tail.store(..., .seq_cst)` and `head.load(.seq_cst)` instead of an explicit fence.

## `std.io` is gone: use `std.fs.File.stdout()` / `std.Io`

- **Symptom**
  - `error: root source file struct 'std' has no member named 'io'`
- **Fix / pattern**
  - For simple stdout printing, use `std.fs.File.stdout()` with a buffered writer:

```zig
var buf: [0x400]u8 = undefined;
var w = std.fs.File.stdout().writer(&buf);
const out = &w.interface;
try out.print("hello {s}\n", .{"world"});
try out.flush();
```

## `std.DynLib.lookup` + function pointers: avoid `@ptrCast`, use `@ptrFromInt`

- **Symptom**
  - Calling a C function pointer obtained from `std.DynLib.lookup(*const fn(...) callconv(.c) ...)` returns nonsense (e.g. `NaN`) even though:
    - Directly linking and calling the same function works
    - Calling the same `dlopen`/`dlsym` flow from C works
- **Fix / pattern**
  - Lookup as `*anyopaque` and convert via integer:

```zig
const sym = lib.lookup(*anyopaque, "cblas_dgemm") orelse return error.SymbolNotFound;
const fn_ptr: *const fn (...) callconv(.c) void = @ptrFromInt(@intFromPtr(sym));
```

### Important follow-up (OpenBLAS CBLAS)

On Zig 0.16 (stage2), **calling CBLAS function pointers loaded via `dlsym` can still miscompile** when the signature passes scalar floats by value (e.g. `cblas_dgemm`/`cblas_sgemm`), even if you use the `@ptrFromInt` conversion.

**Workaround used in `blazt`:** call the **Fortran BLAS** entrypoints (`dgemm_`/`sgemm_`, or ILP64 `dgemm_64_`/`sgemm_64_`) which pass `alpha`/`beta` by pointer and avoid the miscompile.

## No `std.testing.expectPanic` in Zig 0.16’s stdlib

- **Symptom**
  - `error: root source file struct 'testing' has no member named 'expectPanic'`
- **Source**
  - `std/testing.zig` (Zig 0.16.0-dev) contains `expectEqual`, `expectError`, etc., but **does not provide** an `expectPanic` helper.
- **Fix / pattern**
  - Prefer **error returns** for invalid inputs you want to test (then use `std.testing.expectError(...)`).
  - Keep `std.debug.assert` for internal invariants, but don’t unit-test the panic path directly (or test it via a separate process-level harness if you really need it).
