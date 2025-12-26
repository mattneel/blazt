const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const matrix = @import("matrix.zig");
const errors = @import("errors.zig");
const simd = @import("simd.zig");
const gemm_mod = @import("gemm.zig");
const memory = @import("memory.zig");
const build_options = @import("build_options");

extern fn blazt_nt_memcpy(dst: [*]u8, src: [*]const u8, n: usize) callconv(.c) void;

pub const ops = struct {
    // Level 1 (stubs; implemented incrementally)
    /// Copy `n` elements from `x` into `y`.
    ///
    /// - **No allocations**
    /// - **Bounds are asserted** (`n <= x.len` and `n <= y.len`)
    /// - **Overlap policy**: behaves like `memmove` (overlap-safe)
    pub fn copy(comptime T: type, n: usize, x: []const T, y: []T) void {
        if (@sizeOf(T) == 0 or n == 0) return;

        std.debug.assert(n <= x.len);
        std.debug.assert(n <= y.len);

        const src = x[0..n];
        const dst = y[0..n];

        // Quick-out for identical ranges.
        if (@intFromPtr(src.ptr) == @intFromPtr(dst.ptr)) return;

        const bytes = std.math.mul(usize, n, @sizeOf(T)) catch unreachable;
        const src_addr = @intFromPtr(src.ptr);
        const dst_addr = @intFromPtr(dst.ptr);

        const overlap = (dst_addr < src_addr + bytes) and (src_addr < dst_addr + bytes);

        if (overlap and dst_addr > src_addr) {
            std.mem.copyBackwards(T, dst, src);
        } else {
            if (comptime build_options.nt_stores and builtin.cpu.arch == .x86_64) {
                if (!overlap and bytes >= build_options.nt_store_min_bytes) {
                    if (copyNonTemporalBytes(T, dst_addr, src, dst)) return;
                }
            }
            std.mem.copyForwards(T, dst, src);
        }
    }

    /// Attempt a non-temporal (streaming) forward copy for **non-overlapping** ranges.
    ///
    /// Returns `true` if the copy was performed using non-temporal stores, `false` if the caller
    /// should fall back to a regular copy.
    fn copyNonTemporalBytes(comptime T: type, dst_addr: usize, src: []const T, dst: []T) bool {
        // Only bother for trivially bit-copyable types (same policy as memcopy).
        if (comptime !std.meta.hasUniqueRepresentation(T)) return false;

        // Stream stores operate on bytes.
        const byte_len: usize = std.math.mul(usize, src.len, @sizeOf(T)) catch return false;
        if (byte_len == 0) return true;

        const src_bytes: [*]const u8 = @ptrCast(src.ptr);
        const dst_bytes: [*]u8 = @ptrCast(dst.ptr);

        // Our x86 implementation stream-stores 16B chunks; require 16B alignment to be safe.
        const align_bytes: usize = 16;
        if ((@intFromPtr(src_bytes) % align_bytes) != 0) return false;
        if ((dst_addr % align_bytes) != 0) return false;

        blazt_nt_memcpy(dst_bytes, src_bytes, byte_len);
        return true;
    }

    /// Scale `n` elements of `x` in-place: `x[i] = alpha * x[i]`.
    ///
    /// - **No allocations**
    /// - **Bounds are asserted** (`n <= x.len`)
    /// - **SIMD fast path** (vector blocks + scalar tail) when recommended by the target
    pub fn scal(comptime T: type, n: usize, alpha: T, x: []T) void {
        if (@sizeOf(T) == 0 or n == 0) return;

        std.debug.assert(n <= x.len);

        // Safe early-out (preserves NaN/Inf semantics).
        if (alpha == @as(T, 1)) return;

        if (simd.suggestVectorLength(T)) |vl_c| {
            const VL: usize = @intCast(vl_c);
            if (VL > 1) {
                const Vec = @Vector(VL, T);
                const alpha_v: Vec = @splat(alpha);

                var i: usize = 0;
                const vec_end = n - (n % VL);
                while (i < vec_end) : (i += VL) {
                    var chunk: [VL]T = x[i..][0..VL].*;
                    var v: Vec = @bitCast(chunk);
                    v *= alpha_v;
                    chunk = @bitCast(v);
                    x[i..][0..VL].* = chunk;
                }

                while (i < n) : (i += 1) {
                    x[i] = alpha * x[i];
                }
                return;
            }
        }

        // Scalar fallback.
        for (x[0..n]) |*xi| {
            xi.* = alpha * xi.*;
        }
    }

    /// Swap `n` elements between `x` and `y` in-place.
    ///
    /// - **No allocations**
    /// - **Bounds are asserted** (`n <= x.len` and `n <= y.len`)
    /// - **Aliasing/overlap-safe**: if the ranges overlap, we fall back to a scalar per-element swap.
    /// - **SIMD block path** (vector blocks + scalar tail) when the ranges do not overlap.
    pub fn swap(comptime T: type, n: usize, x: []T, y: []T) void {
        if (@sizeOf(T) == 0 or n == 0) return;

        std.debug.assert(n <= x.len);
        std.debug.assert(n <= y.len);

        const xs = x[0..n];
        const ys = y[0..n];

        // Quick-out for identical ranges.
        if (@intFromPtr(xs.ptr) == @intFromPtr(ys.ptr)) return;

        // Detect overlap for the active ranges.
        const bytes = std.math.mul(usize, n, @sizeOf(T)) catch unreachable;
        const x_addr = @intFromPtr(xs.ptr);
        const y_addr = @intFromPtr(ys.ptr);
        const overlap = (y_addr < x_addr + bytes) and (x_addr < y_addr + bytes);

        // Overlap-safe path: scalar semantics match `for i { tmp=x[i]; x[i]=y[i]; y[i]=tmp; }`.
        if (overlap) {
            for (0..n) |i| {
                const tmp = xs[i];
                xs[i] = ys[i];
                ys[i] = tmp;
            }
            return;
        }

        if (simd.suggestVectorLength(T)) |vl_c| {
            const VL: usize = @intCast(vl_c);
            if (VL > 1) {
                var i: usize = 0;
                const vec_end = n - (n % VL);
                while (i < vec_end) : (i += VL) {
                    const ax: [VL]T = xs[i..][0..VL].*;
                    const ay: [VL]T = ys[i..][0..VL].*;
                    xs[i..][0..VL].* = ay;
                    ys[i..][0..VL].* = ax;
                }
                while (i < n) : (i += 1) {
                    const tmp = xs[i];
                    xs[i] = ys[i];
                    ys[i] = tmp;
                }
                return;
            }
        }

        // Scalar fallback.
        for (0..n) |i| {
            const tmp = xs[i];
            xs[i] = ys[i];
            ys[i] = tmp;
        }
    }

    pub fn axpy(comptime T: type, n: usize, alpha: T, x: []const T, y: []T) void {
        if (@sizeOf(T) == 0 or n == 0) return;

        std.debug.assert(n <= x.len);
        std.debug.assert(n <= y.len);

        // BLAS-style quick return.
        if (alpha == @as(T, 0)) return;

        const xs = x[0..n];
        const ys = y[0..n];

        // Detect overlap for the active ranges; if they overlap, fall back to scalar loop to avoid aliasing bugs.
        const bytes = std.math.mul(usize, n, @sizeOf(T)) catch unreachable;
        const x_addr = @intFromPtr(xs.ptr);
        const y_addr = @intFromPtr(ys.ptr);
        const overlap = (y_addr < x_addr + bytes) and (x_addr < y_addr + bytes);

        if (overlap) {
            if (alpha == @as(T, 1)) {
                for (0..n) |i| ys[i] = ys[i] + xs[i];
            } else {
                if (comptime @typeInfo(T) == .float) {
                    for (0..n) |i| ys[i] = simd.mulAdd(alpha, xs[i], ys[i]);
                } else {
                    for (0..n) |i| ys[i] = alpha * xs[i] + ys[i];
                }
            }
            return;
        }

        if (simd.suggestVectorLength(T)) |vl_c| {
            const VL: usize = @intCast(vl_c);
            if (VL > 1) {
                const Vec = @Vector(VL, T);
                const alpha_v: Vec = @splat(alpha);

                var i: usize = 0;
                const vec_end = n - (n % VL);
                while (i < vec_end) : (i += VL) {
                    const x_arr: [VL]T = xs[i..][0..VL].*;
                    var y_arr: [VL]T = ys[i..][0..VL].*;

                    const vx: Vec = @bitCast(x_arr);
                    var vy: Vec = @bitCast(y_arr);

                    if (alpha == @as(T, 1)) {
                        vy += vx;
                    } else {
                        if (comptime @typeInfo(T) == .float) {
                            vy = simd.mulAdd(alpha_v, vx, vy);
                        } else {
                            vy = alpha_v * vx + vy;
                        }
                    }

                    y_arr = @bitCast(vy);
                    ys[i..][0..VL].* = y_arr;
                }

                if (alpha == @as(T, 1)) {
                    while (i < n) : (i += 1) ys[i] = ys[i] + xs[i];
                } else {
                    if (comptime @typeInfo(T) == .float) {
                        while (i < n) : (i += 1) ys[i] = simd.mulAdd(alpha, xs[i], ys[i]);
                    } else {
                        while (i < n) : (i += 1) ys[i] = alpha * xs[i] + ys[i];
                    }
                }
                return;
            }
        }

        // Scalar fallback.
        if (alpha == @as(T, 1)) {
            for (0..n) |i| ys[i] = ys[i] + xs[i];
        } else {
            if (comptime @typeInfo(T) == .float) {
                for (0..n) |i| ys[i] = simd.mulAdd(alpha, xs[i], ys[i]);
            } else {
                for (0..n) |i| ys[i] = alpha * xs[i] + ys[i];
            }
        }
    }

    pub fn dot(comptime T: type, x: []const T, y: []const T) T {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.dot currently only supports floating-point types");
        }

        std.debug.assert(x.len == y.len);
        const n = x.len;
        if (n == 0) return @as(T, 0);

        // For f32, accumulate in f64 with blocked SIMD + Kahan to keep error bounded on large-n.
        if (comptime T == f32) {
            const Acc = f64;
            var sum: Acc = 0.0;
            var c: Acc = 0.0;

            const kahanAddAcc = struct {
                fn add(s: *Acc, comp: *Acc, v: Acc) void {
                    const yk = v - comp.*;
                    const t = s.* + yk;
                    comp.* = (t - s.*) - yk;
                    s.* = t;
                }
            }.add;

            if (simd.suggestVectorLength(f32)) |vl_c| {
                const VL: usize = @intCast(vl_c);
                if (VL > 1 and n >= VL) {
                    const Vec = @Vector(VL, f32);

                    var i: usize = 0;
                    const vec_end = n - (n % VL);
                    while (i < vec_end) : (i += VL) {
                        const vx: Vec = x[i..][0..VL].*;
                        const vy: Vec = y[i..][0..VL].*;
                        const prod: Vec = vx * vy;
                        const block_sum: f32 = @reduce(.Add, prod);
                        kahanAddAcc(&sum, &c, @as(Acc, @floatCast(block_sum)));
                    }

                    while (i < n) : (i += 1) {
                        const prod = @as(Acc, @floatCast(x[i])) * @as(Acc, @floatCast(y[i]));
                        kahanAddAcc(&sum, &c, prod);
                    }

                    return @floatCast(sum);
                }
            }

            // Scalar f64 Kahan fallback.
            for (0..n) |i| {
                const prod = @as(Acc, @floatCast(x[i])) * @as(Acc, @floatCast(y[i]));
                kahanAddAcc(&sum, &c, prod);
            }
            return @floatCast(sum);
        }

        // Generic scalar Kahan for other float types (e.g. f64).
        var sum: T = 0;
        var c: T = 0;
        const kahanAdd = struct {
            fn add(comptime U: type, s: *U, comp: *U, v: U) void {
                const yk = v - comp.*;
                const t = s.* + yk;
                comp.* = (t - s.*) - yk;
                s.* = t;
            }
        }.add;

        if (simd.suggestVectorLength(T)) |vl_c| {
            const VL: usize = @intCast(vl_c);
            if (VL > 1 and n >= VL) {
                const Vec = @Vector(VL, T);
                var acc: Vec = @splat(@as(T, 0));

                var i: usize = 0;
                const vec_end = n - (n % VL);
                while (i < vec_end) : (i += VL) {
                    const vx: Vec = x[i..][0..VL].*;
                    const vy: Vec = y[i..][0..VL].*;
                    acc = simd.mulAdd(vx, vy, acc);
                }

                const acc_arr: [VL]T = @bitCast(acc);
                inline for (acc_arr) |lane| {
                    kahanAdd(T, &sum, &c, lane);
                }

                while (i < n) : (i += 1) {
                    kahanAdd(T, &sum, &c, x[i] * y[i]);
                }

                return sum;
            }
        }

        for (0..n) |i| kahanAdd(T, &sum, &c, x[i] * y[i]);
        return sum;
    }

    pub fn nrm2(comptime T: type, x: []const T) T {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.nrm2 currently only supports floating-point types");
        }

        const n = x.len;
        if (n == 0) return @as(T, 0);

        // Stable scaled sum-of-squares (LAPACK xLASSQ pattern):
        // avoid overflow/underflow when squaring large/small magnitudes.
        if (comptime T == f32) {
            const Acc = f64;
            var scale: Acc = 0.0;
            var ssq: Acc = 1.0;

            for (x) |xi| {
                const ax = @abs(@as(Acc, @floatCast(xi)));
                if (ax != 0.0) {
                    if (scale < ax) {
                        const r = scale / ax;
                        ssq = 1.0 + ssq * (r * r);
                        scale = ax;
                    } else {
                        const r = ax / scale;
                        ssq += r * r;
                    }
                }
            }

            if (scale == 0.0) return @as(T, 0);
            return @floatCast(scale * std.math.sqrt(ssq));
        }

        var scale: T = 0;
        var ssq: T = 1;
        for (x) |xi| {
            const ax = @abs(xi);
            if (ax != @as(T, 0)) {
                if (scale < ax) {
                    const r = scale / ax;
                    ssq = @as(T, 1) + ssq * (r * r);
                    scale = ax;
                } else {
                    const r = ax / scale;
                    ssq += r * r;
                }
            }
        }

        if (scale == @as(T, 0)) return @as(T, 0);
        return scale * std.math.sqrt(ssq);
    }

    pub fn asum(comptime T: type, x: []const T) T {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.asum currently only supports floating-point types");
        }

        const n = x.len;
        if (n == 0) return @as(T, 0);

        if (comptime T == f32) {
            const Acc = f64;
            var sum: Acc = 0.0;
            var c: Acc = 0.0;

            const kahanAddAcc = struct {
                fn add(s: *Acc, comp: *Acc, v: Acc) void {
                    const yk = v - comp.*;
                    const t = s.* + yk;
                    comp.* = (t - s.*) - yk;
                    s.* = t;
                }
            }.add;

            if (simd.suggestVectorLength(f32)) |vl_c| {
                const VL: usize = @intCast(vl_c);
                if (VL > 1 and n >= VL) {
                    const Vec = @Vector(VL, f32);

                    var i: usize = 0;
                    const vec_end = n - (n % VL);
                    while (i < vec_end) : (i += VL) {
                        const vx: Vec = x[i..][0..VL].*;
                        const abs_v: Vec = @abs(vx);
                        const block_sum: f32 = @reduce(.Add, abs_v);
                        kahanAddAcc(&sum, &c, @as(Acc, @floatCast(block_sum)));
                    }
                    while (i < n) : (i += 1) {
                        kahanAddAcc(&sum, &c, @abs(@as(Acc, @floatCast(x[i]))));
                    }
                    return @floatCast(sum);
                }
            }

            for (x) |xi| kahanAddAcc(&sum, &c, @abs(@as(Acc, @floatCast(xi))));
            return @floatCast(sum);
        }

        var sum: T = 0;
        var c: T = 0;
        const kahanAdd = struct {
            fn add(comptime U: type, s: *U, comp: *U, v: U) void {
                const yk = v - comp.*;
                const t = s.* + yk;
                comp.* = (t - s.*) - yk;
                s.* = t;
            }
        }.add;

        for (x) |xi| kahanAdd(T, &sum, &c, @abs(xi));
        return sum;
    }

    // Level 2
    pub fn gemv(
        comptime T: type,
        comptime layout: matrix.Layout,
        trans: types.Trans,
        m: usize,
        n: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        x: []const T,
        beta: T,
        y: []T,
    ) void {
        std.debug.assert(m == a.rows);
        std.debug.assert(n == a.cols);

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };

        const out_len: usize = if (eff_trans == .no_trans) m else n;
        const x_req: usize = if (eff_trans == .no_trans) n else m;

        std.debug.assert(x_req <= x.len);
        std.debug.assert(out_len <= y.len);

        const xv = x[0..x_req];
        const yv = y[0..out_len];

        // Scale y by beta (beta=0 is a special case: do not read y).
        if (beta == @as(T, 0)) {
            @memset(yv, @as(T, 0));
        } else if (beta != @as(T, 1)) {
            for (yv) |*yi| yi.* *= beta;
        }

        if (alpha == @as(T, 0)) return;

        switch (layout) {
            .row_major => switch (eff_trans) {
                .no_trans => {
                    // y[i] += alpha * dot(A[i, :], x)
                    for (0..m) |i| {
                        var sum: T = 0;
                        const row_off = i * a.stride;
                        for (0..n) |j| {
                            sum += a.data[row_off + j] * xv[j];
                        }
                        yv[i] += alpha * sum;
                    }
                },
                .trans => {
                    // y[j] += alpha * dot(A[:, j], x)  (implemented as row-sweep for row-major)
                    for (0..m) |i| {
                        const tmp = alpha * xv[i];
                        const row_off = i * a.stride;
                        for (0..n) |j| {
                            yv[j] += tmp * a.data[row_off + j];
                        }
                    }
                },
                else => unreachable,
            },
            .col_major => switch (eff_trans) {
                .no_trans => {
                    // y += alpha * A * x (column sweep is cache-friendly for col-major)
                    for (0..n) |j| {
                        const tmp = alpha * xv[j];
                        const col_off = j * a.stride;
                        for (0..m) |i| {
                            yv[i] += tmp * a.data[col_off + i];
                        }
                    }
                },
                .trans => {
                    // y[j] += alpha * dot(A[:, j], x)  (column dot is cache-friendly for col-major)
                    for (0..n) |j| {
                        var sum: T = 0;
                        const col_off = j * a.stride;
                        for (0..m) |i| {
                            sum += a.data[col_off + i] * xv[i];
                        }
                        yv[j] += alpha * sum;
                    }
                },
                else => unreachable,
            },
        }
    }

    pub fn ger(
        comptime T: type,
        comptime layout: matrix.Layout,
        m: usize,
        n: usize,
        alpha: T,
        x: []const T,
        y: []const T,
        a: *matrix.Matrix(T, layout),
    ) void {
        std.debug.assert(m == a.rows);
        std.debug.assert(n == a.cols);
        std.debug.assert(m <= x.len);
        std.debug.assert(n <= y.len);

        if (alpha == @as(T, 0)) return;

        const xv = x[0..m];
        const yv = y[0..n];

        switch (layout) {
            .row_major => {
                for (0..m) |i| {
                    const tmp = alpha * xv[i];
                    const row_off = i * a.stride;
                    for (0..n) |j| {
                        a.data[row_off + j] += tmp * yv[j];
                    }
                }
            },
            .col_major => {
                for (0..n) |j| {
                    const tmp = alpha * yv[j];
                    const col_off = j * a.stride;
                    for (0..m) |i| {
                        a.data[col_off + i] += tmp * xv[i];
                    }
                }
            },
        }
    }

    /// Triangular matrix-vector multiply (in-place): `x := op(A) * x`.
    ///
    /// - `A` is `n x n` and triangular per `uplo`
    /// - `diag == .unit` treats diagonal as 1 (diagonal elements are not read)
    /// - `trans == .conj_trans` is treated as `.trans` for real types
    pub fn trmv(
        comptime T: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        trans: types.Trans,
        diag: types.Diag,
        n: usize,
        a: matrix.Matrix(T, layout),
        x: []T,
    ) void {
        std.debug.assert(n == a.rows);
        std.debug.assert(n == a.cols);
        std.debug.assert(n <= x.len);

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };
        const op_uplo: types.UpLo = if (eff_trans == .no_trans)
            uplo
        else
            switch (uplo) {
                .upper => .lower,
                .lower => .upper,
            };

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const aOp = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize, tr: types.Trans) T {
                return if (tr == .no_trans) aAt(mat, i, j) else aAt(mat, j, i);
            }
        }.f;

        const xv = x[0..n];

        switch (op_uplo) {
            .upper => {
                // op(A) is upper: process rows i ascending so x[j>i] are still original.
                for (0..n) |i| {
                    var tmp: T = if (diag == .unit) xv[i] else aOp(a, i, i, eff_trans) * xv[i];
                    for (i + 1..n) |j| {
                        tmp += aOp(a, i, j, eff_trans) * xv[j];
                    }
                    xv[i] = tmp;
                }
            },
            .lower => {
                // op(A) is lower: process rows i descending so x[j<i] are still original.
                var ii: usize = n;
                while (ii != 0) {
                    ii -= 1;
                    const i = ii;
                    var tmp: T = if (diag == .unit) xv[i] else aOp(a, i, i, eff_trans) * xv[i];
                    var j: usize = 0;
                    while (j < i) : (j += 1) {
                        tmp += aOp(a, i, j, eff_trans) * xv[j];
                    }
                    xv[i] = tmp;
                }
            },
        }
    }

    /// Triangular solve (in-place): solve `op(A) * x = b` where `x` is `b` on input.
    ///
    /// - `diag == .unit` treats diagonal as 1 (diagonal elements are not read)
    /// - If `diag == .non_unit` and a diagonal entry is 0, returns `error.Singular`.
    /// - `trans == .conj_trans` is treated as `.trans` for real types
    pub fn trsv(
        comptime T: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        trans: types.Trans,
        diag: types.Diag,
        n: usize,
        a: matrix.Matrix(T, layout),
        x: []T,
    ) errors.TrsvError!void {
        std.debug.assert(n == a.rows);
        std.debug.assert(n == a.cols);
        std.debug.assert(n <= x.len);

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };
        const op_uplo: types.UpLo = if (eff_trans == .no_trans)
            uplo
        else
            switch (uplo) {
                .upper => .lower,
                .lower => .upper,
            };

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const aOp = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize, tr: types.Trans) T {
                return if (tr == .no_trans) aAt(mat, i, j) else aAt(mat, j, i);
            }
        }.f;

        const xv = x[0..n];
        if (n == 0) return;

        switch (op_uplo) {
            .upper => {
                // Back substitution.
                var ii: usize = n;
                while (ii != 0) {
                    ii -= 1;
                    const i = ii;
                    var tmp: T = xv[i];
                    for (i + 1..n) |j| {
                        tmp -= aOp(a, i, j, eff_trans) * xv[j];
                    }
                    if (diag == .non_unit) {
                        const d = aOp(a, i, i, eff_trans);
                        if (d == @as(T, 0)) return error.Singular;
                        tmp /= d;
                    }
                    xv[i] = tmp;
                }
            },
            .lower => {
                // Forward substitution.
                for (0..n) |i| {
                    var tmp: T = xv[i];
                    var j: usize = 0;
                    while (j < i) : (j += 1) {
                        tmp -= aOp(a, i, j, eff_trans) * xv[j];
                    }
                    if (diag == .non_unit) {
                        const d = aOp(a, i, i, eff_trans);
                        if (d == @as(T, 0)) return error.Singular;
                        tmp /= d;
                    }
                    xv[i] = tmp;
                }
            },
        }
    }

    /// Triangular matrix-matrix multiply (in-place): `B := alpha * op(A) * B` or `B := alpha * B * op(A)`.
    ///
    /// - `side == .left`:  `B` is `m×n`, `A` is `m×m`
    /// - `side == .right`: `B` is `m×n`, `A` is `n×n`
    /// - `A` is triangular per `uplo`
    /// - `diag == .unit` treats diagonal as 1 (diagonal elements are not read)
    /// - `trans == .conj_trans` is treated as `.trans` for real types
    pub fn trmm(
        comptime T: type,
        comptime layout: matrix.Layout,
        side: types.Side,
        uplo: types.UpLo,
        trans: types.Trans,
        diag: types.Diag,
        m: usize,
        n: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        b: *matrix.Matrix(T, layout),
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.trmm currently only supports floating-point types");
        }

        std.debug.assert(m == b.rows);
        std.debug.assert(n == b.cols);

        switch (layout) {
            .row_major => {
                std.debug.assert(a.stride >= a.cols);
                std.debug.assert(b.stride >= b.cols);
                if (a.rows != 0 and a.cols != 0) std.debug.assert((a.rows - 1) * a.stride + a.cols <= a.data.len);
                if (b.rows != 0 and b.cols != 0) std.debug.assert((b.rows - 1) * b.stride + b.cols <= b.data.len);
            },
            .col_major => {
                std.debug.assert(a.stride >= a.rows);
                std.debug.assert(b.stride >= b.rows);
                if (a.rows != 0 and a.cols != 0) std.debug.assert((a.cols - 1) * a.stride + a.rows <= a.data.len);
                if (b.rows != 0 and b.cols != 0) std.debug.assert((b.cols - 1) * b.stride + b.rows <= b.data.len);
            },
        }

        switch (side) {
            .left => {
                std.debug.assert(a.rows == m and a.cols == m);
            },
            .right => {
                std.debug.assert(a.rows == n and a.cols == n);
            },
        }

        if (m == 0 or n == 0) return;

        // alpha==0 => B := 0 (avoid reading A or B)
        if (alpha == @as(T, 0)) {
            switch (layout) {
                .row_major => {
                    for (0..m) |i| {
                        const off = i * b.stride;
                        memory.memsetZeroBytes(std.mem.sliceAsBytes(b.data[off .. off + n]));
                    }
                },
                .col_major => {
                    for (0..n) |j| {
                        const off = j * b.stride;
                        memory.memsetZeroBytes(std.mem.sliceAsBytes(b.data[off .. off + m]));
                    }
                },
            }
            return;
        }

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };
        const op_uplo: types.UpLo = if (eff_trans == .no_trans)
            uplo
        else
            switch (uplo) {
                .upper => .lower,
                .lower => .upper,
            };

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const aOp = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize, tr: types.Trans) T {
                return if (tr == .no_trans) aAt(mat, i, j) else aAt(mat, j, i);
            }
        }.f;
        const bAt = struct {
            inline fn f(mat: *matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const bAtPtr = struct {
            inline fn f(mat: *matrix.Matrix(T, layout), i: usize, j: usize) *T {
                return switch (layout) {
                    .row_major => &mat.data[i * mat.stride + j],
                    .col_major => &mat.data[j * mat.stride + i],
                };
            }
        }.f;

        switch (side) {
            .left => {
                // B := alpha * op(A) * B
                for (0..n) |j| {
                    switch (op_uplo) {
                        .upper => {
                            for (0..m) |i| {
                                var tmp: T = if (diag == .unit) bAt(b, i, j) else aOp(a, i, i, eff_trans) * bAt(b, i, j);
                                for (i + 1..m) |k| {
                                    tmp += aOp(a, i, k, eff_trans) * bAt(b, k, j);
                                }
                                bAtPtr(b, i, j).* = alpha * tmp;
                            }
                        },
                        .lower => {
                            var ii: usize = m;
                            while (ii != 0) {
                                ii -= 1;
                                const i = ii;
                                var tmp: T = if (diag == .unit) bAt(b, i, j) else aOp(a, i, i, eff_trans) * bAt(b, i, j);
                                var k: usize = 0;
                                while (k < i) : (k += 1) {
                                    tmp += aOp(a, i, k, eff_trans) * bAt(b, k, j);
                                }
                                bAtPtr(b, i, j).* = alpha * tmp;
                            }
                        },
                    }
                }
            },
            .right => {
                // B := alpha * B * op(A)
                for (0..m) |i| {
                    switch (op_uplo) {
                        .upper => {
                            var jj: usize = n;
                            while (jj != 0) {
                                jj -= 1;
                                const j = jj;
                                var tmp: T = if (diag == .unit) bAt(b, i, j) else bAt(b, i, j) * aOp(a, j, j, eff_trans);
                                var k: usize = 0;
                                while (k < j) : (k += 1) {
                                    tmp += bAt(b, i, k) * aOp(a, k, j, eff_trans);
                                }
                                bAtPtr(b, i, j).* = alpha * tmp;
                            }
                        },
                        .lower => {
                            for (0..n) |j| {
                                var tmp: T = if (diag == .unit) bAt(b, i, j) else bAt(b, i, j) * aOp(a, j, j, eff_trans);
                                for (j + 1..n) |k| {
                                    tmp += bAt(b, i, k) * aOp(a, k, j, eff_trans);
                                }
                                bAtPtr(b, i, j).* = alpha * tmp;
                            }
                        },
                    }
                }
            },
        }
    }

    /// Triangular solve with multiple RHS (in-place): solve `op(A) * X = alpha*B` or `X * op(A) = alpha*B`.
    ///
    /// - `side == .left`:  `B` is `m×n`, `A` is `m×m`, solves `op(A) * X = alpha*B`
    /// - `side == .right`: `B` is `m×n`, `A` is `n×n`, solves `X * op(A) = alpha*B`
    /// - `A` is triangular per `uplo`
    /// - `diag == .unit` treats diagonal as 1 (diagonal elements are not read)
    /// - If `diag == .non_unit` and a diagonal entry is 0, returns `error.Singular`.
    /// - `trans == .conj_trans` is treated as `.trans` for real types
    /// - If `alpha == 0`, sets `B := 0` and returns without checking singularity.
    pub fn trsm(
        comptime T: type,
        comptime layout: matrix.Layout,
        side: types.Side,
        uplo: types.UpLo,
        trans: types.Trans,
        diag: types.Diag,
        m: usize,
        n: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        b: *matrix.Matrix(T, layout),
    ) errors.TrsmError!void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.trsm currently only supports floating-point types");
        }

        std.debug.assert(m == b.rows);
        std.debug.assert(n == b.cols);

        switch (layout) {
            .row_major => {
                std.debug.assert(a.stride >= a.cols);
                std.debug.assert(b.stride >= b.cols);
                if (a.rows != 0 and a.cols != 0) std.debug.assert((a.rows - 1) * a.stride + a.cols <= a.data.len);
                if (b.rows != 0 and b.cols != 0) std.debug.assert((b.rows - 1) * b.stride + b.cols <= b.data.len);
            },
            .col_major => {
                std.debug.assert(a.stride >= a.rows);
                std.debug.assert(b.stride >= b.rows);
                if (a.rows != 0 and a.cols != 0) std.debug.assert((a.cols - 1) * a.stride + a.rows <= a.data.len);
                if (b.rows != 0 and b.cols != 0) std.debug.assert((b.cols - 1) * b.stride + b.rows <= b.data.len);
            },
        }

        const a_dim: usize = switch (side) {
            .left => m,
            .right => n,
        };
        std.debug.assert(a.rows == a_dim and a.cols == a_dim);

        if (m == 0 or n == 0) return;

        // Scale B by alpha (alpha=0 avoids reading B).
        if (alpha == @as(T, 0)) {
            switch (layout) {
                .row_major => {
                    for (0..m) |i| {
                        const off = i * b.stride;
                        memory.memsetZeroBytes(std.mem.sliceAsBytes(b.data[off .. off + n]));
                    }
                },
                .col_major => {
                    for (0..n) |j| {
                        const off = j * b.stride;
                        memory.memsetZeroBytes(std.mem.sliceAsBytes(b.data[off .. off + m]));
                    }
                },
            }
            return;
        } else if (alpha != @as(T, 1)) {
            switch (layout) {
                .row_major => {
                    for (0..m) |i| {
                        const off = i * b.stride;
                        for (0..n) |j| b.data[off + j] *= alpha;
                    }
                },
                .col_major => {
                    for (0..n) |j| {
                        const off = j * b.stride;
                        for (0..m) |i| b.data[off + i] *= alpha;
                    }
                },
            }
        }

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };
        const op_uplo: types.UpLo = if (eff_trans == .no_trans)
            uplo
        else
            switch (uplo) {
                .upper => .lower,
                .lower => .upper,
            };

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const aOp = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize, tr: types.Trans) T {
                return if (tr == .no_trans) aAt(mat, i, j) else aAt(mat, j, i);
            }
        }.f;
        const bAt = struct {
            inline fn f(mat: *matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const bAtPtr = struct {
            inline fn f(mat: *matrix.Matrix(T, layout), i: usize, j: usize) *T {
                return switch (layout) {
                    .row_major => &mat.data[i * mat.stride + j],
                    .col_major => &mat.data[j * mat.stride + i],
                };
            }
        }.f;

        switch (side) {
            .left => {
                // Solve op(A) * X = B, column by column.
                for (0..n) |j| {
                    switch (op_uplo) {
                        .upper => {
                            // Back substitution.
                            var ii: usize = m;
                            while (ii != 0) {
                                ii -= 1;
                                const i = ii;
                                var tmp: T = bAt(b, i, j);
                                for (i + 1..m) |k| {
                                    tmp -= aOp(a, i, k, eff_trans) * bAt(b, k, j);
                                }
                                if (diag == .non_unit) {
                                    const d = aOp(a, i, i, eff_trans);
                                    if (d == @as(T, 0)) return error.Singular;
                                    tmp /= d;
                                }
                                bAtPtr(b, i, j).* = tmp;
                            }
                        },
                        .lower => {
                            // Forward substitution.
                            for (0..m) |i| {
                                var tmp: T = bAt(b, i, j);
                                var k: usize = 0;
                                while (k < i) : (k += 1) {
                                    tmp -= aOp(a, i, k, eff_trans) * bAt(b, k, j);
                                }
                                if (diag == .non_unit) {
                                    const d = aOp(a, i, i, eff_trans);
                                    if (d == @as(T, 0)) return error.Singular;
                                    tmp /= d;
                                }
                                bAtPtr(b, i, j).* = tmp;
                            }
                        },
                    }
                }
            },
            .right => {
                // Solve X * op(A) = B, row by row.
                for (0..m) |i| {
                    switch (op_uplo) {
                        .upper => {
                            // Forward substitution over columns.
                            for (0..n) |j| {
                                var tmp: T = bAt(b, i, j);
                                var k: usize = 0;
                                while (k < j) : (k += 1) {
                                    tmp -= bAt(b, i, k) * aOp(a, k, j, eff_trans);
                                }
                                if (diag == .non_unit) {
                                    const d = aOp(a, j, j, eff_trans);
                                    if (d == @as(T, 0)) return error.Singular;
                                    tmp /= d;
                                }
                                bAtPtr(b, i, j).* = tmp;
                            }
                        },
                        .lower => {
                            // Back substitution over columns.
                            var jj: usize = n;
                            while (jj != 0) {
                                jj -= 1;
                                const j = jj;
                                var tmp: T = bAt(b, i, j);
                                for (j + 1..n) |k| {
                                    tmp -= bAt(b, i, k) * aOp(a, k, j, eff_trans);
                                }
                                if (diag == .non_unit) {
                                    const d = aOp(a, j, j, eff_trans);
                                    if (d == @as(T, 0)) return error.Singular;
                                    tmp /= d;
                                }
                                bAtPtr(b, i, j).* = tmp;
                            }
                        },
                    }
                }
            },
        }
    }

    /// Symmetric matrix-vector multiply: `y := alpha*A*x + beta*y`.
    ///
    /// Only the triangle indicated by `uplo` is referenced; the other triangle is ignored.
    pub fn symv(
        comptime T: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        n: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        x: []const T,
        beta: T,
        y: []T,
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.symv currently only supports floating-point types");
        }

        std.debug.assert(n == a.rows);
        std.debug.assert(n == a.cols);
        std.debug.assert(n <= x.len);
        std.debug.assert(n <= y.len);

        const xv = x[0..n];
        const yv = y[0..n];

        // Scale y by beta (beta=0 avoids reading y).
        if (beta == @as(T, 0)) {
            @memset(yv, @as(T, 0));
        } else if (beta != @as(T, 1)) {
            for (yv) |*yi| yi.* *= beta;
        }

        if (alpha == @as(T, 0) or n == 0) return;

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;

        switch (uplo) {
            .upper => {
                for (0..n) |i| {
                    const tmp1 = alpha * xv[i];
                    var tmp2: T = 0;

                    var j: usize = 0;
                    while (j < i) : (j += 1) {
                        const aji = aAt(a, j, i);
                        yv[j] += tmp1 * aji;
                        tmp2 += aji * xv[j];
                    }

                    yv[i] += tmp1 * aAt(a, i, i) + alpha * tmp2;
                }
            },
            .lower => {
                for (0..n) |i| {
                    const tmp1 = alpha * xv[i];
                    var tmp2: T = 0;

                    yv[i] += tmp1 * aAt(a, i, i);

                    for (i + 1..n) |j| {
                        const aji = aAt(a, j, i);
                        yv[j] += tmp1 * aji;
                        tmp2 += aji * xv[j];
                    }

                    yv[i] += alpha * tmp2;
                }
            },
        }
    }

    /// Hermitian matrix-vector multiply: `y := alpha*A*x + beta*y`.
    ///
    /// - Only the triangle indicated by `uplo` is referenced; the other triangle is ignored.
    /// - Off-diagonal elements are used with conjugation per Hermitian symmetry.
    /// - Diagonal elements are assumed real; imaginary parts are ignored.
    pub fn hemv(
        comptime C: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        n: usize,
        alpha: C,
        a: matrix.Matrix(C, layout),
        x: []const C,
        beta: C,
        y: []C,
    ) void {
        if (!@hasDecl(C, "init") or !@hasDecl(C, "add") or !@hasDecl(C, "mul") or !@hasDecl(C, "conjugate")) {
            @compileError("ops.hemv expects a std.math.Complex(T) type");
        }
        if (!@hasField(C, "re") or !@hasField(C, "im")) {
            @compileError("ops.hemv expects a std.math.Complex(T) type");
        }

        const R = @TypeOf(@as(C, undefined).re);
        const zero: C = C.init(@as(R, 0), @as(R, 0));

        std.debug.assert(n == a.rows);
        std.debug.assert(n == a.cols);
        std.debug.assert(n <= x.len);
        std.debug.assert(n <= y.len);

        const xv = x[0..n];
        const yv = y[0..n];

        const beta_is_zero = (beta.re == @as(R, 0)) and (beta.im == @as(R, 0));
        const beta_is_one = (beta.re == @as(R, 1)) and (beta.im == @as(R, 0));
        if (beta_is_zero) {
            for (yv) |*yi| yi.* = zero;
        } else if (!beta_is_one) {
            for (yv) |*yi| yi.* = yi.*.mul(beta);
        }

        const alpha_is_zero = (alpha.re == @as(R, 0)) and (alpha.im == @as(R, 0));
        if (alpha_is_zero or n == 0) return;

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(C, layout), i: usize, j: usize) C {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;

        const diagReal = struct {
            inline fn f(v: C) C {
                return C.init(v.re, @as(R, 0));
            }
        }.f;

        switch (uplo) {
            .upper => {
                for (0..n) |i| {
                    const tmp1 = alpha.mul(xv[i]);
                    var tmp2: C = zero;

                    var j: usize = 0;
                    while (j < i) : (j += 1) {
                        const aji = aAt(a, j, i);
                        yv[j] = yv[j].add(tmp1.mul(aji));
                        tmp2 = tmp2.add(aji.conjugate().mul(xv[j]));
                    }

                    const di = diagReal(aAt(a, i, i));
                    yv[i] = yv[i].add(tmp1.mul(di));
                    yv[i] = yv[i].add(alpha.mul(tmp2));
                }
            },
            .lower => {
                for (0..n) |i| {
                    const tmp1 = alpha.mul(xv[i]);
                    var tmp2: C = zero;

                    const di = diagReal(aAt(a, i, i));
                    yv[i] = yv[i].add(tmp1.mul(di));

                    for (i + 1..n) |j| {
                        const aji = aAt(a, j, i);
                        yv[j] = yv[j].add(tmp1.mul(aji));
                        tmp2 = tmp2.add(aji.conjugate().mul(xv[j]));
                    }

                    yv[i] = yv[i].add(alpha.mul(tmp2));
                }
            },
        }
    }

    // Level 3
    pub fn gemm(
        comptime T: type,
        comptime layout: matrix.Layout,
        trans_a: types.Trans,
        trans_b: types.Trans,
        alpha: T,
        a: matrix.Matrix(T, layout),
        b: matrix.Matrix(T, layout),
        beta: T,
        c: *matrix.Matrix(T, layout),
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.gemm currently only supports floating-point types");
        }

        // For real types, `.conj_trans` is equivalent to `.trans`.
        const eff_a: types.Trans = switch (trans_a) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };
        const eff_b: types.Trans = switch (trans_b) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };

        const m: usize = c.rows;
        const n: usize = c.cols;

        // Validate storage invariants for strided matrices (debug only).
        switch (layout) {
            .row_major => {
                std.debug.assert(a.stride >= a.cols);
                std.debug.assert(b.stride >= b.cols);
                std.debug.assert(c.stride >= c.cols);
                if (a.rows != 0 and a.cols != 0) std.debug.assert((a.rows - 1) * a.stride + a.cols <= a.data.len);
                if (b.rows != 0 and b.cols != 0) std.debug.assert((b.rows - 1) * b.stride + b.cols <= b.data.len);
                if (c.rows != 0 and c.cols != 0) std.debug.assert((c.rows - 1) * c.stride + c.cols <= c.data.len);
            },
            .col_major => {
                std.debug.assert(a.stride >= a.rows);
                std.debug.assert(b.stride >= b.rows);
                std.debug.assert(c.stride >= c.rows);
                if (a.rows != 0 and a.cols != 0) std.debug.assert((a.cols - 1) * a.stride + a.rows <= a.data.len);
                if (b.rows != 0 and b.cols != 0) std.debug.assert((b.cols - 1) * b.stride + b.rows <= b.data.len);
                if (c.rows != 0 and c.cols != 0) std.debug.assert((c.cols - 1) * c.stride + c.rows <= c.data.len);
            },
        }

        // Determine k and validate dimensions against transposes:
        // op(A) is m×k, op(B) is k×n, C is m×n.
        const k: usize = blk: {
            if (eff_a == .no_trans) {
                std.debug.assert(a.rows == m);
                break :blk a.cols;
            } else {
                std.debug.assert(a.cols == m);
                break :blk a.rows;
            }
        };

        if (eff_b == .no_trans) {
            std.debug.assert(b.rows == k);
            std.debug.assert(b.cols == n);
        } else {
            std.debug.assert(b.cols == k);
            std.debug.assert(b.rows == n);
        }

        if (m == 0 or n == 0) return;

        // alpha==0 or k==0 => C := beta*C (beta=0 avoids reading C)
        if (k == 0 or alpha == @as(T, 0)) {
            if (beta == @as(T, 0)) {
                switch (layout) {
                    .row_major => {
                        for (0..m) |i| {
                            const row_off = i * c.stride;
                            memory.memsetZeroBytes(std.mem.sliceAsBytes(c.data[row_off .. row_off + n]));
                        }
                    },
                    .col_major => {
                        for (0..n) |j| {
                            const col_off = j * c.stride;
                            memory.memsetZeroBytes(std.mem.sliceAsBytes(c.data[col_off .. col_off + m]));
                        }
                    },
                }
            } else if (beta != @as(T, 1)) {
                switch (layout) {
                    .row_major => {
                        for (0..m) |i| {
                            const row_off = i * c.stride;
                            for (0..n) |j| c.data[row_off + j] *= beta;
                        }
                    },
                    .col_major => {
                        for (0..n) |j| {
                            const col_off = j * c.stride;
                            for (0..m) |i| c.data[col_off + i] *= beta;
                        }
                    },
                }
            }
            return;
        }

        // Fast path: use the cache-blocked kernel for the common case.
        if (eff_a == .no_trans and eff_b == .no_trans) {
            const P: gemm_mod.TileParams = comptime gemm_mod.computeTileParams(T);
            var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
            var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;

            switch (layout) {
                .col_major => {
                    gemm_mod.gemmBlocked(T, m, n, k, alpha, a, b, beta, c, pack_a[0..], pack_b[0..]);
                    return;
                },
                .row_major => {
                    // Row-major GEMM can be expressed as a col-major GEMM on transposed views:
                    // (A*B)^T = B^T * A^T
                    const a_t: matrix.Matrix(T, .col_major) = .{
                        .data = a.data,
                        .rows = a.cols,
                        .cols = a.rows,
                        .stride = a.stride,
                        .allocator = a.allocator,
                    };
                    const b_t: matrix.Matrix(T, .col_major) = .{
                        .data = b.data,
                        .rows = b.cols,
                        .cols = b.rows,
                        .stride = b.stride,
                        .allocator = b.allocator,
                    };
                    var c_t: matrix.Matrix(T, .col_major) = .{
                        .data = c.data,
                        .rows = c.cols,
                        .cols = c.rows,
                        .stride = c.stride,
                        .allocator = c.allocator,
                    };

                    gemm_mod.gemmBlocked(T, n, m, k, alpha, b_t, a_t, beta, &c_t, pack_a[0..], pack_b[0..]);
                    return;
                },
            }
        }

        // Fallback: generic triple loop supporting transposes and stride.
        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const bAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;

        switch (layout) {
            .row_major => {
                switch (eff_a) {
                    .no_trans => switch (eff_b) {
                        .no_trans => {
                            for (0..m) |i| {
                                const row_off = i * c.stride;
                                for (0..n) |j| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, i, p) * bAt(b, p, j);
                                    const idx = row_off + j;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        .trans => {
                            for (0..m) |i| {
                                const row_off = i * c.stride;
                                for (0..n) |j| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, i, p) * bAt(b, j, p);
                                    const idx = row_off + j;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        else => unreachable,
                    },
                    .trans => switch (eff_b) {
                        .no_trans => {
                            for (0..m) |i| {
                                const row_off = i * c.stride;
                                for (0..n) |j| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, p, i) * bAt(b, p, j);
                                    const idx = row_off + j;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        .trans => {
                            for (0..m) |i| {
                                const row_off = i * c.stride;
                                for (0..n) |j| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, p, i) * bAt(b, j, p);
                                    const idx = row_off + j;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        else => unreachable,
                    },
                    else => unreachable,
                }
            },
            .col_major => {
                switch (eff_a) {
                    .no_trans => switch (eff_b) {
                        .no_trans => {
                            for (0..n) |j| {
                                const col_off = j * c.stride;
                                for (0..m) |i| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, i, p) * bAt(b, p, j);
                                    const idx = col_off + i;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        .trans => {
                            for (0..n) |j| {
                                const col_off = j * c.stride;
                                for (0..m) |i| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, i, p) * bAt(b, j, p);
                                    const idx = col_off + i;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        else => unreachable,
                    },
                    .trans => switch (eff_b) {
                        .no_trans => {
                            for (0..n) |j| {
                                const col_off = j * c.stride;
                                for (0..m) |i| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, p, i) * bAt(b, p, j);
                                    const idx = col_off + i;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        .trans => {
                            for (0..n) |j| {
                                const col_off = j * c.stride;
                                for (0..m) |i| {
                                    var sum: T = @as(T, 0);
                                    for (0..k) |p| sum += aAt(a, p, i) * bAt(b, j, p);
                                    const idx = col_off + i;
                                    if (beta == @as(T, 0)) {
                                        c.data[idx] = alpha * sum;
                                    } else if (beta == @as(T, 1)) {
                                        c.data[idx] = alpha * sum + c.data[idx];
                                    } else {
                                        c.data[idx] = alpha * sum + beta * c.data[idx];
                                    }
                                }
                            }
                        },
                        else => unreachable,
                    },
                    else => unreachable,
                }
            },
        }
    }

    /// Symmetric rank-k update: `C := alpha*op(A)*op(A)^T + beta*C`.
    ///
    /// - `C` is `n×n`.
    /// - If `trans == .no_trans`, `A` is `n×k`; otherwise `A` is `k×n` and `op(A)=A^T`.
    /// - Only the triangle indicated by `uplo` is updated; the other triangle is left unchanged.
    fn scaleTriangleInPlace(
        comptime T: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        n: usize,
        beta: T,
        c: *matrix.Matrix(T, layout),
    ) void {
        if (beta == @as(T, 1)) return;
        const beta_is_zero = (beta == @as(T, 0));

        switch (uplo) {
            .upper => {
                for (0..n) |j| {
                    for (0..j + 1) |i| {
                        if (beta_is_zero) {
                            c.atPtr(i, j).* = @as(T, 0);
                        } else {
                            c.atPtr(i, j).* *= beta;
                        }
                    }
                }
            },
            .lower => {
                for (0..n) |j| {
                    for (j..n) |i| {
                        if (beta_is_zero) {
                            c.atPtr(i, j).* = @as(T, 0);
                        } else {
                            c.atPtr(i, j).* *= beta;
                        }
                    }
                }
            },
        }
    }

    pub fn syrk(
        comptime T: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        trans: types.Trans,
        n: usize,
        k: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        beta: T,
        c: *matrix.Matrix(T, layout),
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.syrk currently only supports floating-point types");
        }

        // For real types, `.conj_trans` is equivalent to `.trans`.
        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };

        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);
        if (eff_trans == .no_trans) {
            std.debug.assert(a.rows == n);
            std.debug.assert(a.cols == k);
        } else {
            std.debug.assert(a.rows == k);
            std.debug.assert(a.cols == n);
        }

        if (n == 0) return;

        if (k == 0 or alpha == @as(T, 0)) {
            // Only scale the referenced triangle.
            scaleTriangleInPlace(T, layout, uplo, n, beta, c);
            return;
        }

        const uplo_swapped: types.UpLo = switch (uplo) {
            .upper => .lower,
            .lower => .upper,
        };
        const flipTrans: types.Trans = if (eff_trans == .no_trans) .trans else .no_trans;

        switch (layout) {
            .col_major => syrkBlockedColMajor(T, uplo, eff_trans, n, k, alpha, a, beta, c),
            .row_major => {
                // Map row-major to col-major on transposed views.
                // Since the result is symmetric, updating `C^T` with flipped `uplo` is equivalent.
                const a_t: matrix.Matrix(T, .col_major) = .{
                    .data = a.data,
                    .rows = a.cols,
                    .cols = a.rows,
                    .stride = a.stride,
                    .allocator = a.allocator,
                };
                var c_t: matrix.Matrix(T, .col_major) = .{
                    .data = c.data,
                    .rows = c.cols,
                    .cols = c.rows,
                    .stride = c.stride,
                    .allocator = c.allocator,
                };
                syrkBlockedColMajor(T, uplo_swapped, flipTrans, n, k, alpha, a_t, beta, &c_t);
            },
        }
    }

    fn syrkBlockedColMajor(
        comptime T: type,
        uplo: types.UpLo,
        trans_eff: types.Trans, // .no_trans or .trans
        n: usize,
        k: usize,
        alpha: T,
        a: matrix.Matrix(T, .col_major),
        beta: T,
        c: *matrix.Matrix(T, .col_major),
    ) void {
        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);

        // Transpose view of A (whole-matrix; alignment preserved).
        const a_t: matrix.Matrix(T, .row_major) = .{
            .data = a.data,
            .rows = a.cols,
            .cols = a.rows,
            .stride = a.stride,
            .allocator = a.allocator,
        };

        const P: gemm_mod.TileParams = comptime gemm_mod.computeTileParams(T);
        const MR: usize = P.MR;
        const NR: usize = P.NR;
        const KC: usize = P.KC;
        const MC: usize = P.MC;
        const NC: usize = P.NC;

        // Sanity for our packing buffers.
        var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;

        const rs_c: usize = 1;
        const cs_c: usize = c.stride;

        var jc: usize = 0;
        while (jc < n) : (jc += NC) {
            const nc_cur = @min(NC, n - jc);

            var pc: usize = 0;
            while (pc < k) : (pc += KC) {
                const kc_cur = @min(KC, k - pc);
                const beta_block: T = if (pc == 0) beta else @as(T, 1);

                var jr: usize = 0;
                while (jr < nc_cur) : (jr += NR) {
                    const nr_cur = @min(NR, nc_cur - jr);
                    const col0 = jc + jr;

                    // Pack one KC×NR micro-panel from the "right" operand.
                    const pb = pack_b[0 .. NR * kc_cur];
                    if (trans_eff == .no_trans) {
                        // B = A^T (row-major view).
                        gemm_mod.packB(T, .row_major, NR, kc_cur, nr_cur, a_t, pc, col0, pb);
                    } else {
                        // B = A (col-major).
                        gemm_mod.packB(T, .col_major, NR, kc_cur, nr_cur, a, pc, col0, pb);
                    }

                    var ic: usize = 0;
                    while (ic < n) : (ic += MC) {
                        if (uplo == .upper and ic >= jc + nc_cur) break; // entirely below diag for this column panel

                        const mc_cur = @min(MC, n - ic);

                        var ir: usize = 0;
                        while (ir < mc_cur) : (ir += MR) {
                            const mr_cur = @min(MR, mc_cur - ir);
                            const row0 = ic + ir;

                            // Quick block-level skip/full classification.
                            const fully_above = (row0 + mr_cur <= col0);
                            const fully_below = (row0 >= col0 + nr_cur);
                            const do_full: bool = switch (uplo) {
                                .upper => fully_above,
                                .lower => fully_below,
                            };
                            const do_skip: bool = switch (uplo) {
                                .upper => fully_below,
                                .lower => fully_above,
                            };
                            if (do_skip) continue;

                            // Pack one MR×KC micro-panel from the "left" operand.
                            const pa = pack_a[0 .. MR * kc_cur];
                            if (trans_eff == .no_trans) {
                                // A = A (col-major).
                                gemm_mod.packA(T, .col_major, MR, kc_cur, mr_cur, a, row0, pc, pa);
                            } else {
                                // A = A^T (row-major view).
                                gemm_mod.packA(T, .row_major, MR, kc_cur, mr_cur, a_t, row0, pc, pa);
                            }

                            if (do_full) {
                                const c_ptr: [*]T = c.data[col0 * c.stride + row0 ..].ptr;
                                gemm_mod.microKernelPartial(
                                    T,
                                    MR,
                                    NR,
                                    kc_cur,
                                    pa.ptr,
                                    pb.ptr,
                                    c_ptr,
                                    rs_c,
                                    cs_c,
                                    alpha,
                                    beta_block,
                                    mr_cur,
                                    nr_cur,
                                );
                                continue;
                            }

                            // Overlaps diagonal: compute into a small temp and scatter only the requested triangle.
                            var tmp: [MR * NR]T = undefined;
                            for (0..nr_cur) |jj| {
                                for (0..mr_cur) |ii| {
                                    tmp[jj * MR + ii] = c.data[(col0 + jj) * c.stride + (row0 + ii)];
                                }
                            }

                            gemm_mod.microKernelPartial(
                                T,
                                MR,
                                NR,
                                kc_cur,
                                pa.ptr,
                                pb.ptr,
                                tmp[0..].ptr,
                                1,
                                MR,
                                alpha,
                                beta_block,
                                mr_cur,
                                nr_cur,
                            );

                            for (0..nr_cur) |jj| {
                                for (0..mr_cur) |ii| {
                                    const gi = row0 + ii;
                                    const gj = col0 + jj;
                                    const keep = switch (uplo) {
                                        .upper => gi <= gj,
                                        .lower => gi >= gj,
                                    };
                                    if (keep) {
                                        c.data[gj * c.stride + gi] = tmp[jj * MR + ii];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn triGemmUpdateColMajorSquare(
        comptime T: type,
        uplo: types.UpLo,
        n: usize,
        k: usize,
        alpha: T,
        comptime layout_a: matrix.Layout,
        a: matrix.Matrix(T, layout_a), // n×k
        comptime layout_b: matrix.Layout,
        b: matrix.Matrix(T, layout_b), // k×n
        beta: T,
        c: *matrix.Matrix(T, .col_major), // n×n
    ) void {
        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);
        std.debug.assert(a.rows == n);
        std.debug.assert(a.cols == k);
        std.debug.assert(b.rows == k);
        std.debug.assert(b.cols == n);

        const P: gemm_mod.TileParams = comptime gemm_mod.computeTileParams(T);
        const MR: usize = P.MR;
        const NR: usize = P.NR;
        const KC: usize = P.KC;
        const MC: usize = P.MC;
        const NC: usize = P.NC;

        var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;

        const rs_c: usize = 1;
        const cs_c: usize = c.stride;

        var jc: usize = 0;
        while (jc < n) : (jc += NC) {
            const nc_cur = @min(NC, n - jc);

            var pc: usize = 0;
            while (pc < k) : (pc += KC) {
                const kc_cur = @min(KC, k - pc);
                const beta_block: T = if (pc == 0) beta else @as(T, 1);

                var jr: usize = 0;
                while (jr < nc_cur) : (jr += NR) {
                    const nr_cur = @min(NR, nc_cur - jr);
                    const col0 = jc + jr;

                    const pb = pack_b[0 .. NR * kc_cur];
                    gemm_mod.packB(T, layout_b, NR, kc_cur, nr_cur, b, pc, col0, pb);

                    var ic: usize = 0;
                    while (ic < n) : (ic += MC) {
                        if (uplo == .upper and ic >= jc + nc_cur) break;

                        const mc_cur = @min(MC, n - ic);

                        var ir: usize = 0;
                        while (ir < mc_cur) : (ir += MR) {
                            const mr_cur = @min(MR, mc_cur - ir);
                            const row0 = ic + ir;

                            const fully_above = (row0 + mr_cur <= col0);
                            const fully_below = (row0 >= col0 + nr_cur);
                            const do_full: bool = switch (uplo) {
                                .upper => fully_above,
                                .lower => fully_below,
                            };
                            const do_skip: bool = switch (uplo) {
                                .upper => fully_below,
                                .lower => fully_above,
                            };
                            if (do_skip) continue;

                            const pa = pack_a[0 .. MR * kc_cur];
                            gemm_mod.packA(T, layout_a, MR, kc_cur, mr_cur, a, row0, pc, pa);

                            if (do_full) {
                                const c_ptr: [*]T = c.data[col0 * c.stride + row0 ..].ptr;
                                gemm_mod.microKernelPartial(
                                    T,
                                    MR,
                                    NR,
                                    kc_cur,
                                    pa.ptr,
                                    pb.ptr,
                                    c_ptr,
                                    rs_c,
                                    cs_c,
                                    alpha,
                                    beta_block,
                                    mr_cur,
                                    nr_cur,
                                );
                                continue;
                            }

                            var tmp: [MR * NR]T = undefined;
                            for (0..nr_cur) |jj| {
                                for (0..mr_cur) |ii| {
                                    tmp[jj * MR + ii] = c.data[(col0 + jj) * c.stride + (row0 + ii)];
                                }
                            }

                            gemm_mod.microKernelPartial(
                                T,
                                MR,
                                NR,
                                kc_cur,
                                pa.ptr,
                                pb.ptr,
                                tmp[0..].ptr,
                                1,
                                MR,
                                alpha,
                                beta_block,
                                mr_cur,
                                nr_cur,
                            );

                            for (0..nr_cur) |jj| {
                                for (0..mr_cur) |ii| {
                                    const gi = row0 + ii;
                                    const gj = col0 + jj;
                                    const keep = switch (uplo) {
                                        .upper => gi <= gj,
                                        .lower => gi >= gj,
                                    };
                                    if (keep) {
                                        c.data[gj * c.stride + gi] = tmp[jj * MR + ii];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn syr2kColMajor(
        comptime T: type,
        uplo: types.UpLo,
        trans_eff: types.Trans, // .no_trans or .trans
        n: usize,
        k: usize,
        alpha: T,
        a: matrix.Matrix(T, .col_major),
        b: matrix.Matrix(T, .col_major),
        beta: T,
        c: *matrix.Matrix(T, .col_major),
    ) void {
        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);

        // Whole-matrix transpose views (alignment preserved).
        const a_t: matrix.Matrix(T, .row_major) = .{
            .data = a.data,
            .rows = a.cols,
            .cols = a.rows,
            .stride = a.stride,
            .allocator = a.allocator,
        };
        const b_t: matrix.Matrix(T, .row_major) = .{
            .data = b.data,
            .rows = b.cols,
            .cols = b.rows,
            .stride = b.stride,
            .allocator = b.allocator,
        };

        if (trans_eff == .no_trans) {
            // A,B are n×k.
            triGemmUpdateColMajorSquare(T, uplo, n, k, alpha, .col_major, a, .row_major, b_t, beta, c);
            triGemmUpdateColMajorSquare(T, uplo, n, k, alpha, .col_major, b, .row_major, a_t, @as(T, 1), c);
        } else {
            // A,B are k×n; op(A)=A^T and op(B)=B^T.
            triGemmUpdateColMajorSquare(T, uplo, n, k, alpha, .row_major, a_t, .col_major, b, beta, c);
            triGemmUpdateColMajorSquare(T, uplo, n, k, alpha, .row_major, b_t, .col_major, a, @as(T, 1), c);
        }
    }

    /// Symmetric rank-2k update: `C := alpha*op(A)*op(B)^T + alpha*op(B)*op(A)^T + beta*C`.
    ///
    /// - `C` is `n×n`.
    /// - If `trans == .no_trans`, `A` and `B` are `n×k`; otherwise `A` and `B` are `k×n` and `op(X)=X^T`.
    /// - Only the triangle indicated by `uplo` is updated; the other triangle is left unchanged.
    pub fn syr2k(
        comptime T: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        trans: types.Trans,
        n: usize,
        k: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        b: matrix.Matrix(T, layout),
        beta: T,
        c: *matrix.Matrix(T, layout),
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.syr2k currently only supports floating-point types");
        }

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };

        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);
        if (eff_trans == .no_trans) {
            std.debug.assert(a.rows == n and a.cols == k);
            std.debug.assert(b.rows == n and b.cols == k);
        } else {
            std.debug.assert(a.rows == k and a.cols == n);
            std.debug.assert(b.rows == k and b.cols == n);
        }

        if (n == 0) return;

        if (k == 0 or alpha == @as(T, 0)) {
            scaleTriangleInPlace(T, layout, uplo, n, beta, c);
            return;
        }

        const uplo_swapped: types.UpLo = switch (uplo) {
            .upper => .lower,
            .lower => .upper,
        };
        const flipTrans: types.Trans = if (eff_trans == .no_trans) .trans else .no_trans;

        switch (layout) {
            .col_major => syr2kColMajor(T, uplo, eff_trans, n, k, alpha, a, b, beta, c),
            .row_major => {
                const a_t: matrix.Matrix(T, .col_major) = .{
                    .data = a.data,
                    .rows = a.cols,
                    .cols = a.rows,
                    .stride = a.stride,
                    .allocator = a.allocator,
                };
                const b_t: matrix.Matrix(T, .col_major) = .{
                    .data = b.data,
                    .rows = b.cols,
                    .cols = b.rows,
                    .stride = b.stride,
                    .allocator = b.allocator,
                };
                var c_t: matrix.Matrix(T, .col_major) = .{
                    .data = c.data,
                    .rows = c.cols,
                    .cols = c.rows,
                    .stride = c.stride,
                    .allocator = c.allocator,
                };
                syr2kColMajor(T, uplo_swapped, flipTrans, n, k, alpha, a_t, b_t, beta, &c_t);
            },
        }
    }

    /// Hermitian rank-2k update: `C := alpha*op(A)*op(B)^H + conj(alpha)*op(B)*op(A)^H + beta*C`.
    ///
    /// - `C` is `n×n` Hermitian; only the triangle `uplo` is updated.
    /// - `beta` is real (BLAS-compatible). `alpha` is complex.
    /// - `trans` is `.no_trans` (A,B are `n×k`) or `.conj_trans` (A,B are `k×n`). `.trans` is treated as `.conj_trans`.
    /// - Diagonal elements are forced real (imag part set to 0).
    pub fn her2k(
        comptime C: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        trans: types.Trans,
        n: usize,
        k: usize,
        alpha: C,
        a: matrix.Matrix(C, layout),
        b: matrix.Matrix(C, layout),
        beta: @TypeOf(@as(C, undefined).re),
        c: *matrix.Matrix(C, layout),
    ) void {
        if (!@hasDecl(C, "init") or !@hasDecl(C, "add") or !@hasDecl(C, "mul") or !@hasDecl(C, "conjugate")) {
            @compileError("ops.her2k expects a std.math.Complex(T) type");
        }
        if (!@hasField(C, "re") or !@hasField(C, "im")) {
            @compileError("ops.her2k expects a std.math.Complex(T) type");
        }

        const R = @TypeOf(@as(C, undefined).re);
        const zero: C = C.init(@as(R, 0), @as(R, 0));
        const beta_c: C = C.init(beta, @as(R, 0));
        const alpha_conj: C = alpha.conjugate();

        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .conj_trans, .trans => .conj_trans,
        };

        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);
        if (eff_trans == .no_trans) {
            std.debug.assert(a.rows == n and a.cols == k);
            std.debug.assert(b.rows == n and b.cols == k);
        } else {
            std.debug.assert(a.rows == k and a.cols == n);
            std.debug.assert(b.rows == k and b.cols == n);
        }

        if (n == 0) return;

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(C, layout), i: usize, j: usize) C {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const cAtPtr = struct {
            inline fn f(mat: *matrix.Matrix(C, layout), i: usize, j: usize) *C {
                return switch (layout) {
                    .row_major => &mat.data[i * mat.stride + j],
                    .col_major => &mat.data[j * mat.stride + i],
                };
            }
        }.f;

        const alpha_is_zero = (alpha.re == @as(R, 0)) and (alpha.im == @as(R, 0));
        if (k == 0 or alpha_is_zero) {
            if (beta == @as(R, 1)) return;
            const beta_is_zero = beta == @as(R, 0);
            switch (uplo) {
                .upper => for (0..n) |j| for (0..j + 1) |i| {
                    const p = cAtPtr(c, i, j);
                    if (beta_is_zero) {
                        p.* = if (i == j) C.init(@as(R, 0), @as(R, 0)) else zero;
                    } else {
                        p.* = p.*.mul(beta_c);
                        if (i == j) p.* = C.init(p.*.re, @as(R, 0));
                    }
                },
                .lower => for (0..n) |j| for (j..n) |i| {
                    const p = cAtPtr(c, i, j);
                    if (beta_is_zero) {
                        p.* = if (i == j) C.init(@as(R, 0), @as(R, 0)) else zero;
                    } else {
                        p.* = p.*.mul(beta_c);
                        if (i == j) p.* = C.init(p.*.re, @as(R, 0));
                    }
                },
            }
            return;
        }

        const beta_is_zero = beta == @as(R, 0);

        switch (uplo) {
            .upper => {
                for (0..n) |j| {
                    for (0..j + 1) |i| {
                        var sum1: C = zero;
                        var sum2: C = zero;
                        if (eff_trans == .no_trans) {
                            // sum1 = Σ_p A[i,p] * conj(B[j,p])
                            // sum2 = Σ_p B[i,p] * conj(A[j,p])
                            for (0..k) |p| {
                                sum1 = sum1.add(aAt(a, i, p).mul(aAt(b, j, p).conjugate()));
                                sum2 = sum2.add(aAt(b, i, p).mul(aAt(a, j, p).conjugate()));
                            }
                        } else {
                            // sum1 = Σ_p conj(A[p,i]) * B[p,j]
                            // sum2 = Σ_p conj(B[p,i]) * A[p,j]
                            for (0..k) |p| {
                                sum1 = sum1.add(aAt(a, p, i).conjugate().mul(aAt(b, p, j)));
                                sum2 = sum2.add(aAt(b, p, i).conjugate().mul(aAt(a, p, j)));
                            }
                        }

                        const cp = cAtPtr(c, i, j);
                        var out: C = alpha.mul(sum1).add(alpha_conj.mul(sum2));
                        if (!beta_is_zero) out = out.add(cp.*.mul(beta_c));
                        cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
                    }
                }
            },
            .lower => {
                for (0..n) |j| {
                    for (j..n) |i| {
                        var sum1: C = zero;
                        var sum2: C = zero;
                        if (eff_trans == .no_trans) {
                            for (0..k) |p| {
                                sum1 = sum1.add(aAt(a, i, p).mul(aAt(b, j, p).conjugate()));
                                sum2 = sum2.add(aAt(b, i, p).mul(aAt(a, j, p).conjugate()));
                            }
                        } else {
                            for (0..k) |p| {
                                sum1 = sum1.add(aAt(a, p, i).conjugate().mul(aAt(b, p, j)));
                                sum2 = sum2.add(aAt(b, p, i).conjugate().mul(aAt(a, p, j)));
                            }
                        }

                        const cp = cAtPtr(c, i, j);
                        var out: C = alpha.mul(sum1).add(alpha_conj.mul(sum2));
                        if (!beta_is_zero) out = out.add(cp.*.mul(beta_c));
                        cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
                    }
                }
            },
        }
    }

    fn flipUpLo(uplo: types.UpLo) types.UpLo {
        return switch (uplo) {
            .upper => .lower,
            .lower => .upper,
        };
    }

    fn flipSide(side: types.Side) types.Side {
        return switch (side) {
            .left => .right,
            .right => .left,
        };
    }

    fn symmAt(
        comptime T: type,
        comptime layout: matrix.Layout,
        a: matrix.Matrix(T, layout),
        uplo: types.UpLo,
        i: usize,
        j: usize,
    ) T {
        const aAt = struct {
            inline fn f(mat: matrix.Matrix(T, layout), ii: usize, jj: usize) T {
                return switch (layout) {
                    .row_major => mat.data[ii * mat.stride + jj],
                    .col_major => mat.data[jj * mat.stride + ii],
                };
            }
        }.f;

        if (i == j) return aAt(a, i, j);
        return switch (uplo) {
            .upper => if (i < j) aAt(a, i, j) else aAt(a, j, i),
            .lower => if (i > j) aAt(a, i, j) else aAt(a, j, i),
        };
    }

    fn packASymm(
        comptime T: type,
        comptime MR: usize,
        kc: usize,
        m: usize,
        a: matrix.Matrix(T, .col_major),
        uplo: types.UpLo,
        row0: usize,
        col0: usize,
        dst: []align(memory.CacheLine) T,
    ) void {
        std.debug.assert(MR > 0);
        std.debug.assert(m <= MR);
        std.debug.assert(row0 + m <= a.rows);
        std.debug.assert(col0 + kc <= a.cols);
        std.debug.assert(dst.len >= MR * kc);
        std.debug.assert(@intFromPtr(dst.ptr) % memory.CacheLine == 0);

        for (0..kc) |p| {
            inline for (0..MR) |ii| {
                dst[p * MR + ii] = if (ii < m) symmAt(T, .col_major, a, uplo, row0 + ii, col0 + p) else @as(T, 0);
            }
        }
    }

    fn packBSymm(
        comptime T: type,
        comptime NR: usize,
        kc: usize,
        n: usize,
        a: matrix.Matrix(T, .col_major),
        uplo: types.UpLo,
        row0: usize,
        col0: usize,
        dst: []align(memory.CacheLine) T,
    ) void {
        std.debug.assert(NR > 0);
        std.debug.assert(n <= NR);
        std.debug.assert(row0 + kc <= a.rows);
        std.debug.assert(col0 + n <= a.cols);
        std.debug.assert(dst.len >= NR * kc);
        std.debug.assert(@intFromPtr(dst.ptr) % memory.CacheLine == 0);

        for (0..kc) |p| {
            inline for (0..NR) |jj| {
                dst[p * NR + jj] = if (jj < n) symmAt(T, .col_major, a, uplo, row0 + p, col0 + jj) else @as(T, 0);
            }
        }
    }

    fn symmColMajor(
        comptime T: type,
        side: types.Side,
        uplo: types.UpLo,
        m: usize,
        n: usize,
        alpha: T,
        a: matrix.Matrix(T, .col_major),
        b: matrix.Matrix(T, .col_major),
        beta: T,
        c: *matrix.Matrix(T, .col_major),
    ) void {
        std.debug.assert(m == c.rows);
        std.debug.assert(n == c.cols);

        // Determine A dimensions based on side.
        switch (side) {
            .left => {
                std.debug.assert(a.rows == m);
                std.debug.assert(a.cols == m);
                std.debug.assert(b.rows == m);
                std.debug.assert(b.cols == n);
            },
            .right => {
                std.debug.assert(a.rows == n);
                std.debug.assert(a.cols == n);
                std.debug.assert(b.rows == m);
                std.debug.assert(b.cols == n);
            },
        }

        if (m == 0 or n == 0) return;

        // alpha==0 => C := beta*C (beta=0 avoids reading C)
        if (alpha == @as(T, 0)) {
            if (beta == @as(T, 0)) {
                for (0..n) |j| {
                    const off = j * c.stride;
                    memory.memsetZeroBytes(std.mem.sliceAsBytes(c.data[off .. off + m]));
                }
            } else if (beta != @as(T, 1)) {
                for (0..n) |j| {
                    const off = j * c.stride;
                    for (0..m) |i| c.data[off + i] *= beta;
                }
            }
            return;
        }

        // Blocked micro-kernel path (float-only).
        const P: gemm_mod.TileParams = comptime gemm_mod.computeTileParams(T);
        const MR: usize = P.MR;
        const NR: usize = P.NR;
        const KC: usize = P.KC;
        const MC: usize = P.MC;
        const NC: usize = P.NC;

        var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;

        const rs_c: usize = 1;
        const cs_c: usize = c.stride;

        const k_total: usize = switch (side) {
            .left => m,
            .right => n,
        };

        var jc: usize = 0;
        while (jc < n) : (jc += NC) {
            const nc_cur = @min(NC, n - jc);

            var pc: usize = 0;
            while (pc < k_total) : (pc += KC) {
                const kc_cur = @min(KC, k_total - pc);
                const beta_block: T = if (pc == 0) beta else @as(T, 1);

                var jr: usize = 0;
                while (jr < nc_cur) : (jr += NR) {
                    const nr_cur = @min(NR, nc_cur - jr);
                    const col0 = jc + jr;

                    // Pack right operand micro-panel.
                    const pb = pack_b[0 .. NR * kc_cur];
                    switch (side) {
                        .left => gemm_mod.packB(T, .col_major, NR, kc_cur, nr_cur, b, pc, col0, pb),
                        .right => packBSymm(T, NR, kc_cur, nr_cur, a, uplo, pc, col0, pb),
                    }

                    var ic: usize = 0;
                    while (ic < m) : (ic += MC) {
                        const mc_cur = @min(MC, m - ic);

                        var ir: usize = 0;
                        while (ir < mc_cur) : (ir += MR) {
                            const mr_cur = @min(MR, mc_cur - ir);
                            const row0 = ic + ir;

                            // Pack left operand micro-panel.
                            const pa = pack_a[0 .. MR * kc_cur];
                            switch (side) {
                                .left => packASymm(T, MR, kc_cur, mr_cur, a, uplo, row0, pc, pa),
                                .right => gemm_mod.packA(T, .col_major, MR, kc_cur, mr_cur, b, row0, pc, pa),
                            }

                            // Update C block at (row0, col0).
                            const c_ptr: [*]T = c.data[col0 * c.stride + row0 ..].ptr;
                            gemm_mod.microKernelPartial(
                                T,
                                MR,
                                NR,
                                kc_cur,
                                pa.ptr,
                                pb.ptr,
                                c_ptr,
                                rs_c,
                                cs_c,
                                alpha,
                                beta_block,
                                mr_cur,
                                nr_cur,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Symmetric matrix-matrix multiply:
    ///
    /// - `side == .left`:  `C := alpha*A*B + beta*C`, with `A` symmetric `m×m`.
    /// - `side == .right`: `C := alpha*B*A + beta*C`, with `A` symmetric `n×n`.
    ///
    /// Only the triangle indicated by `uplo` is referenced; the other triangle is ignored.
    pub fn symm(
        comptime T: type,
        comptime layout: matrix.Layout,
        side: types.Side,
        uplo: types.UpLo,
        m: usize,
        n: usize,
        alpha: T,
        a: matrix.Matrix(T, layout),
        b: matrix.Matrix(T, layout),
        beta: T,
        c: *matrix.Matrix(T, layout),
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.symm currently only supports floating-point types");
        }

        std.debug.assert(m == c.rows);
        std.debug.assert(n == c.cols);
        if (m == 0 or n == 0) return;

        switch (layout) {
            .col_major => symmColMajor(T, side, uplo, m, n, alpha, a, b, beta, c),
            .row_major => {
                // Map row-major to col-major on transposed views:
                // (A*B)^T = B^T*A^T, and A^T is symmetric. This swaps `side`.
                const side_t = flipSide(side);
                const uplo_t = flipUpLo(uplo);

                const a_t: matrix.Matrix(T, .col_major) = .{
                    .data = a.data,
                    .rows = a.cols,
                    .cols = a.rows,
                    .stride = a.stride,
                    .allocator = a.allocator,
                };
                const b_t: matrix.Matrix(T, .col_major) = .{
                    .data = b.data,
                    .rows = b.cols,
                    .cols = b.rows,
                    .stride = b.stride,
                    .allocator = b.allocator,
                };
                var c_t: matrix.Matrix(T, .col_major) = .{
                    .data = c.data,
                    .rows = c.cols,
                    .cols = c.rows,
                    .stride = c.stride,
                    .allocator = c.allocator,
                };

                symmColMajor(T, side_t, uplo_t, n, m, alpha, a_t, b_t, beta, &c_t);
            },
        }
    }

    fn hermAt(
        comptime C: type,
        comptime layout: matrix.Layout,
        a: matrix.Matrix(C, layout),
        uplo: types.UpLo,
        i: usize,
        j: usize,
    ) C {
        const aAt = struct {
            inline fn f(mat: matrix.Matrix(C, layout), ii: usize, jj: usize) C {
                return switch (layout) {
                    .row_major => mat.data[ii * mat.stride + jj],
                    .col_major => mat.data[jj * mat.stride + ii],
                };
            }
        }.f;

        const R = @TypeOf(@as(C, undefined).re);
        const diagReal = struct {
            inline fn f(v: C) C {
                return C.init(v.re, @as(R, 0));
            }
        }.f;

        if (i == j) return diagReal(aAt(a, i, j));
        return switch (uplo) {
            .upper => if (i < j) aAt(a, i, j) else aAt(a, j, i).conjugate(),
            .lower => if (i > j) aAt(a, i, j) else aAt(a, j, i).conjugate(),
        };
    }

    /// Hermitian matrix-matrix multiply:
    ///
    /// - `side == .left`:  `C := alpha*A*B + beta*C`, with `A` Hermitian `m×m`.
    /// - `side == .right`: `C := alpha*B*A + beta*C`, with `A` Hermitian `n×n`.
    ///
    /// Only the triangle indicated by `uplo` is referenced; the other triangle is ignored.
    /// Off-diagonal elements are used with conjugation per Hermitian symmetry; diagonals are treated as real.
    pub fn hemm(
        comptime C: type,
        comptime layout: matrix.Layout,
        side: types.Side,
        uplo: types.UpLo,
        m: usize,
        n: usize,
        alpha: C,
        a: matrix.Matrix(C, layout),
        b: matrix.Matrix(C, layout),
        beta: C,
        c: *matrix.Matrix(C, layout),
    ) void {
        if (!@hasDecl(C, "init") or !@hasDecl(C, "add") or !@hasDecl(C, "mul") or !@hasDecl(C, "conjugate")) {
            @compileError("ops.hemm expects a std.math.Complex(T) type");
        }
        if (!@hasField(C, "re") or !@hasField(C, "im")) {
            @compileError("ops.hemm expects a std.math.Complex(T) type");
        }

        const R = @TypeOf(@as(C, undefined).re);
        const zero: C = C.init(@as(R, 0), @as(R, 0));

        std.debug.assert(m == c.rows);
        std.debug.assert(n == c.cols);

        switch (side) {
            .left => {
                std.debug.assert(a.rows == m and a.cols == m);
                std.debug.assert(b.rows == m and b.cols == n);
            },
            .right => {
                std.debug.assert(a.rows == n and a.cols == n);
                std.debug.assert(b.rows == m and b.cols == n);
            },
        }

        if (m == 0 or n == 0) return;

        const beta_is_zero = (beta.re == @as(R, 0)) and (beta.im == @as(R, 0));
        const beta_is_one = (beta.re == @as(R, 1)) and (beta.im == @as(R, 0));
        if (beta_is_zero) {
            for (c.data) |*v| v.* = zero;
        } else if (!beta_is_one) {
            for (c.data) |*v| v.* = v.*.mul(beta);
        }

        const alpha_is_zero = (alpha.re == @as(R, 0)) and (alpha.im == @as(R, 0));
        if (alpha_is_zero) return;

        const bAt = struct {
            inline fn f(mat: matrix.Matrix(C, layout), i: usize, j: usize) C {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const cAtPtr = struct {
            inline fn f(mat: *matrix.Matrix(C, layout), i: usize, j: usize) *C {
                return switch (layout) {
                    .row_major => &mat.data[i * mat.stride + j],
                    .col_major => &mat.data[j * mat.stride + i],
                };
            }
        }.f;

        switch (side) {
            .left => {
                // C(m×n) += alpha * A(m×m) * B(m×n)
                for (0..n) |j| {
                    for (0..m) |i| {
                        var sum: C = zero;
                        for (0..m) |p| {
                            sum = sum.add(hermAt(C, layout, a, uplo, i, p).mul(bAt(b, p, j)));
                        }
                        const cp = cAtPtr(c, i, j);
                        cp.* = cp.*.add(alpha.mul(sum));
                    }
                }
            },
            .right => {
                // C(m×n) += alpha * B(m×n) * A(n×n)
                for (0..n) |j| {
                    for (0..m) |i| {
                        var sum: C = zero;
                        for (0..n) |p| {
                            sum = sum.add(bAt(b, i, p).mul(hermAt(C, layout, a, uplo, p, j)));
                        }
                        const cp = cAtPtr(c, i, j);
                        cp.* = cp.*.add(alpha.mul(sum));
                    }
                }
            },
        }
    }

    /// Hermitian rank-k update: `C := alpha*op(A)*op(A)^H + beta*C`.
    ///
    /// - `C` is `n×n` Hermitian, stored as a full matrix; only the triangle `uplo` is updated.
    /// - `alpha` and `beta` are real scalars (BLAS-compatible).
    /// - `trans` is `.no_trans` (A is `n×k`) or `.conj_trans` (A is `k×n`). `.trans` is treated as `.conj_trans`.
    /// - Diagonal elements are forced real (imag part set to 0).
    pub fn herk(
        comptime C: type,
        comptime layout: matrix.Layout,
        uplo: types.UpLo,
        trans: types.Trans,
        n: usize,
        k: usize,
        alpha: @TypeOf(@as(C, undefined).re),
        a: matrix.Matrix(C, layout),
        beta: @TypeOf(@as(C, undefined).re),
        c: *matrix.Matrix(C, layout),
    ) void {
        if (!@hasDecl(C, "init") or !@hasDecl(C, "add") or !@hasDecl(C, "mul") or !@hasDecl(C, "conjugate")) {
            @compileError("ops.herk expects a std.math.Complex(T) type");
        }
        if (!@hasField(C, "re") or !@hasField(C, "im")) {
            @compileError("ops.herk expects a std.math.Complex(T) type");
        }

        const R = @TypeOf(@as(C, undefined).re);
        const zero: C = C.init(@as(R, 0), @as(R, 0));
        const alpha_c: C = C.init(alpha, @as(R, 0));
        const beta_c: C = C.init(beta, @as(R, 0));

        // For Hermitian update, `.trans` is not meaningful; treat it as `.conj_trans`.
        const eff_trans: types.Trans = switch (trans) {
            .no_trans => .no_trans,
            .conj_trans, .trans => .conj_trans,
        };

        std.debug.assert(n == c.rows);
        std.debug.assert(n == c.cols);
        if (eff_trans == .no_trans) {
            std.debug.assert(a.rows == n);
            std.debug.assert(a.cols == k);
        } else {
            std.debug.assert(a.rows == k);
            std.debug.assert(a.cols == n);
        }

        if (n == 0) return;

        const aAt = struct {
            inline fn f(mat: matrix.Matrix(C, layout), i: usize, j: usize) C {
                return switch (layout) {
                    .row_major => mat.data[i * mat.stride + j],
                    .col_major => mat.data[j * mat.stride + i],
                };
            }
        }.f;
        const cAtPtr = struct {
            inline fn f(mat: *matrix.Matrix(C, layout), i: usize, j: usize) *C {
                return switch (layout) {
                    .row_major => &mat.data[i * mat.stride + j],
                    .col_major => &mat.data[j * mat.stride + i],
                };
            }
        }.f;

        // Handle alpha==0 or k==0 by scaling only the referenced triangle.
        const alpha_is_zero = (alpha == @as(R, 0));
        if (k == 0 or alpha_is_zero) {
            if (beta == @as(R, 1)) return;
            const beta_is_zero = (beta == @as(R, 0));
            switch (uplo) {
                .upper => for (0..n) |j| for (0..j + 1) |i| {
                    const p = cAtPtr(c, i, j);
                    if (beta_is_zero) {
                        p.* = if (i == j) C.init(@as(R, 0), @as(R, 0)) else zero;
                    } else {
                        p.* = p.*.mul(beta_c);
                        if (i == j) p.* = C.init(p.*.re, @as(R, 0));
                    }
                },
                .lower => for (0..n) |j| for (j..n) |i| {
                    const p = cAtPtr(c, i, j);
                    if (beta_is_zero) {
                        p.* = if (i == j) C.init(@as(R, 0), @as(R, 0)) else zero;
                    } else {
                        p.* = p.*.mul(beta_c);
                        if (i == j) p.* = C.init(p.*.re, @as(R, 0));
                    }
                },
            }
            return;
        }

        const beta_is_zero = (beta == @as(R, 0));

        switch (uplo) {
            .upper => {
                for (0..n) |j| {
                    for (0..j + 1) |i| {
                        var sum: C = zero;
                        if (eff_trans == .no_trans) {
                            // sum = Σ_p A[i,p] * conj(A[j,p])
                            for (0..k) |p| {
                                sum = sum.add(aAt(a, i, p).mul(aAt(a, j, p).conjugate()));
                            }
                        } else {
                            // sum = Σ_p conj(A[p,i]) * A[p,j]
                            for (0..k) |p| {
                                sum = sum.add(aAt(a, p, i).conjugate().mul(aAt(a, p, j)));
                            }
                        }

                        const cp = cAtPtr(c, i, j);
                        const scaled = alpha_c.mul(sum);
                        const out = if (beta_is_zero) scaled else scaled.add(cp.*.mul(beta_c));
                        cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
                    }
                }
            },
            .lower => {
                for (0..n) |j| {
                    for (j..n) |i| {
                        var sum: C = zero;
                        if (eff_trans == .no_trans) {
                            for (0..k) |p| {
                                sum = sum.add(aAt(a, i, p).mul(aAt(a, j, p).conjugate()));
                            }
                        } else {
                            for (0..k) |p| {
                                sum = sum.add(aAt(a, p, i).conjugate().mul(aAt(a, p, j)));
                            }
                        }

                        const cp = cAtPtr(c, i, j);
                        const scaled = alpha_c.mul(sum);
                        const out = if (beta_is_zero) scaled else scaled.add(cp.*.mul(beta_c));
                        cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
                    }
                }
            },
        }
    }

    // LAPACK
    /// LU factorization with partial pivoting (in-place): computes `P*A = L*U`.
    ///
    /// - `a` is overwritten with `L` (strictly lower triangle, unit diagonal implied) and `U` (upper triangle).
    /// - `ipiv` stores pivot indices **1-based** (LAPACK-style):
    ///   - For each `k` in `0..min(m,n)`, a row interchange was performed between row `k` and row `ipiv[k]-1`.
    ///   - Applying these swaps in order to the original matrix yields `P*A`.
    /// - If a pivot is exactly 0, returns `error.Singular` (factorization stops).
    pub fn lu(comptime T: type, a: *matrix.Matrix(T, .row_major), ipiv: []i32) errors.LuError!void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.lu currently only supports floating-point types");
        }

        const m: usize = a.rows;
        const n: usize = a.cols;
        const stride: usize = a.stride;
        std.debug.assert(stride >= n);

        const min_mn: usize = @min(m, n);
        std.debug.assert(ipiv.len >= min_mn);

        if (min_mn == 0) return;

        var data = a.data;

        const nb: usize = 32;
        var j: usize = 0;
        while (j < min_mn) : (j += nb) {
            const jb: usize = @min(nb, min_mn - j);

            // Factor panel A[j..m, j..j+jb) using unblocked LU with partial pivoting.
            var jj: usize = 0;
            while (jj < jb) : (jj += 1) {
                const col: usize = j + jj;

                // Find pivot row in col (max abs value among rows col..m-1).
                var piv: usize = col;
                var best_abs: T = @abs(data[col * stride + col]);
                if (!std.math.isNan(best_abs)) {
                    var r: usize = col + 1;
                    while (r < m) : (r += 1) {
                        const v: T = @abs(data[r * stride + col]);
                        if (std.math.isNan(v)) {
                            piv = r;
                            break;
                        }
                        if (v > best_abs) {
                            best_abs = v;
                            piv = r;
                        }
                    }
                }

                ipiv[col] = @intCast(piv + 1); // 1-based

                if (piv != col) {
                    // Swap entire rows (affects previously computed columns too).
                    const off_a = col * stride;
                    const off_b = piv * stride;
                    var ccol: usize = 0;
                    while (ccol < n) : (ccol += 1) {
                        const tmp = data[off_a + ccol];
                        data[off_a + ccol] = data[off_b + ccol];
                        data[off_b + ccol] = tmp;
                    }
                }

                const pivot: T = data[col * stride + col];
                if (pivot == @as(T, 0)) return error.Singular;

                // Compute multipliers below the diagonal in this column.
                var r: usize = col + 1;
                while (r < m) : (r += 1) {
                    data[r * stride + col] /= pivot;
                }

                // Update the rest of the panel (columns col+1 .. j+jb-1).
                var k: usize = col + 1;
                while (k < j + jb) : (k += 1) {
                    const u: T = data[col * stride + k];
                    var i: usize = col + 1;
                    while (i < m) : (i += 1) {
                        data[i * stride + k] -= data[i * stride + col] * u;
                    }
                }
            }

            const j2: usize = j + jb;

            // Compute block row U12: solve L11 * U12 = A12 in-place (L11 is unit lower).
            if (j2 < n) {
                var k: usize = j2;
                while (k < n) : (k += 1) {
                    var i: usize = 0;
                    while (i < jb) : (i += 1) {
                        var tmp: T = data[(j + i) * stride + k];
                        var p: usize = 0;
                        while (p < i) : (p += 1) {
                            tmp -= data[(j + i) * stride + (j + p)] * data[(j + p) * stride + k];
                        }
                        data[(j + i) * stride + k] = tmp;
                    }
                }
            }

            // Trailing update: A22 -= L21 * U12.
            if (j2 < m and j2 < n) {
                var i: usize = j2;
                while (i < m) : (i += 1) {
                    var k: usize = j2;
                    while (k < n) : (k += 1) {
                        var sum: T = @as(T, 0);
                        var p: usize = 0;
                        while (p < jb) : (p += 1) {
                            sum += data[i * stride + (j + p)] * data[(j + p) * stride + k];
                        }
                        data[i * stride + k] -= sum;
                    }
                }
            }
        }
    }

    pub fn cholesky(comptime T: type, uplo: types.UpLo, a: *matrix.Matrix(T, .row_major)) errors.CholeskyError!void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.cholesky currently only supports floating-point types");
        }

        const n: usize = a.rows;
        std.debug.assert(a.cols == n);
        std.debug.assert(a.stride >= a.cols);
        if (n == 0) return;

        // In-place unblocked Cholesky.
        switch (uplo) {
            .lower => {
                // Compute L such that A = L*L^T. Store L in the lower triangle (unit is not implied).
                for (0..n) |j| {
                    var sum: T = a.data[j * a.stride + j];
                    var k: usize = 0;
                    while (k < j) : (k += 1) {
                        const ljk = a.data[j * a.stride + k];
                        sum -= ljk * ljk;
                    }

                    if (!(sum > @as(T, 0))) return error.NotPositiveDefinite;
                    const ljj: T = @sqrt(sum);
                    a.data[j * a.stride + j] = ljj;

                    var i: usize = j + 1;
                    while (i < n) : (i += 1) {
                        var t: T = a.data[i * a.stride + j];
                        k = 0;
                        while (k < j) : (k += 1) {
                            t -= a.data[i * a.stride + k] * a.data[j * a.stride + k];
                        }
                        a.data[i * a.stride + j] = t / ljj;
                    }
                }
            },
            .upper => {
                // Compute U such that A = U^T*U. Store U in the upper triangle.
                for (0..n) |j| {
                    var sum: T = a.data[j * a.stride + j];
                    var k: usize = 0;
                    while (k < j) : (k += 1) {
                        const ukj = a.data[k * a.stride + j];
                        sum -= ukj * ukj;
                    }

                    if (!(sum > @as(T, 0))) return error.NotPositiveDefinite;
                    const ujj: T = @sqrt(sum);
                    a.data[j * a.stride + j] = ujj;

                    var col: usize = j + 1;
                    while (col < n) : (col += 1) {
                        var t: T = a.data[j * a.stride + col];
                        k = 0;
                        while (k < j) : (k += 1) {
                            t -= a.data[k * a.stride + j] * a.data[k * a.stride + col];
                        }
                        a.data[j * a.stride + col] = t / ujj;
                    }
                }
            },
        }
    }

    /// QR factorization via Householder reflectors (in-place): computes `A = Q*R`.
    ///
    /// - Input `a` is `m×n` (row-major) and is overwritten with:
    ///   - `R` in the upper triangle / upper trapezoid
    ///   - Householder vectors below the diagonal (each reflector has implicit `v[0]=1`)
    /// - `tau` (length >= `min(m,n)`) stores the Householder scalar factors.
    ///
    /// To explicitly form `Q`, use `qrFormQ`.
    pub fn qr(comptime T: type, a: *matrix.Matrix(T, .row_major), tau: []T) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.qr currently only supports floating-point types");
        }

        const m: usize = a.rows;
        const n: usize = a.cols;
        const stride: usize = a.stride;
        std.debug.assert(stride >= n);

        const kmax: usize = @min(m, n);
        std.debug.assert(tau.len >= kmax);
        if (kmax == 0) return;

        var data = a.data;

        var k: usize = 0;
        while (k < kmax) : (k += 1) {
            // Compute Householder reflector to zero out a[k+1..m, k].
            const kk = k * stride + k;
            const alpha: T = data[kk];

            // sigma = ||x_tail||^2
            var sigma: T = @as(T, 0);
            var i: usize = k + 1;
            while (i < m) : (i += 1) {
                const v = data[i * stride + k];
                sigma += v * v;
            }

            if (sigma == @as(T, 0)) {
                tau[k] = @as(T, 0);
                continue;
            }

            const norm: T = @sqrt(alpha * alpha + sigma);
            const sign: T = if (alpha >= @as(T, 0)) @as(T, 1) else @as(T, -1);
            const beta: T = -sign * norm;

            const v1: T = alpha - beta;
            tau[k] = (beta - alpha) / beta;

            // Store beta on diagonal; scale the vector tail in-place (v[0] is implicit 1).
            data[kk] = beta;
            i = k + 1;
            while (i < m) : (i += 1) {
                data[i * stride + k] /= v1;
            }

            // Apply H = I - tau*v*v^T to remaining columns.
            var j: usize = k + 1;
            while (j < n) : (j += 1) {
                // w = v^T * A[k:m, j]
                var w: T = data[k * stride + j];
                i = k + 1;
                while (i < m) : (i += 1) {
                    w += data[i * stride + k] * data[i * stride + j];
                }
                w *= tau[k];

                data[k * stride + j] -= w;
                i = k + 1;
                while (i < m) : (i += 1) {
                    data[i * stride + j] -= data[i * stride + k] * w;
                }
            }
        }
    }

    /// Form an explicit `m×m` orthogonal matrix `Q` from a QR factorization produced by `qr`.
    ///
    /// - `a_qr` is the output of `qr` (same `m×n` shape as the original input).
    /// - `tau` must match the `tau` produced by `qr`.
    /// - `q_out` must be `m×m` (row-major) and will be filled with `Q`.
    pub fn qrFormQ(comptime T: type, a_qr: matrix.Matrix(T, .row_major), tau: []const T, q_out: *matrix.Matrix(T, .row_major)) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.qrFormQ currently only supports floating-point types");
        }

        const m: usize = a_qr.rows;
        const n: usize = a_qr.cols;
        _ = n;
        std.debug.assert(q_out.rows == m);
        std.debug.assert(q_out.cols == m);
        std.debug.assert(q_out.stride >= q_out.cols);

        const kmax: usize = @min(m, a_qr.cols);
        std.debug.assert(tau.len >= kmax);

        // Q := I
        for (0..m) |i| {
            const off = i * q_out.stride;
            for (0..m) |j| q_out.data[off + j] = if (i == j) @as(T, 1) else @as(T, 0);
        }

        if (kmax == 0) return;

        // Apply reflectors from last to first: Q = H_{kmax-1} ... H_0
        var kk: usize = kmax;
        while (kk != 0) {
            kk -= 1;
            const k = kk;
            const tau_k: T = tau[k];
            if (tau_k == @as(T, 0)) continue;

            // v = [1; a_qr[k+1..m, k]]
            var col: usize = 0;
            while (col < m) : (col += 1) {
                var w: T = q_out.data[k * q_out.stride + col];
                var i: usize = k + 1;
                while (i < m) : (i += 1) {
                    w += a_qr.data[i * a_qr.stride + k] * q_out.data[i * q_out.stride + col];
                }
                w *= tau_k;

                q_out.data[k * q_out.stride + col] -= w;
                i = k + 1;
                while (i < m) : (i += 1) {
                    q_out.data[i * q_out.stride + col] -= a_qr.data[i * a_qr.stride + k] * w;
                }
            }
        }
    }

    /// Economy SVD (real): `A = U * diag(S) * Vt` with `k = min(m,n)`.
    ///
    /// - `a` (m×n) is used as workspace and is overwritten.
    /// - `s` length >= `k` receives singular values in **descending** order.
    /// - `u` must be `m×k` (column-orthonormal).
    /// - `vt` must be `k×n` (row-orthonormal).
    ///
    /// Current implementation uses a one-sided Jacobi method.
    pub fn svd(
        comptime T: type,
        a: *matrix.Matrix(T, .row_major),
        s: []T,
        u: *matrix.Matrix(T, .row_major),
        vt: *matrix.Matrix(T, .row_major),
    ) errors.SvdError!void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.svd currently only supports floating-point types");
        }

        const m: usize = a.rows;
        const n: usize = a.cols;
        const k: usize = @min(m, n);

        std.debug.assert(a.stride >= n);
        std.debug.assert(s.len >= k);
        std.debug.assert(u.rows == m and u.cols == k);
        std.debug.assert(vt.rows == k and vt.cols == n);
        std.debug.assert(u.stride >= u.cols);
        std.debug.assert(vt.stride >= vt.cols);

        if (k == 0) return;

        const eps: T = @sqrt(std.math.floatEps(T));
        const max_sweeps: usize = 64;

        const absT = struct {
            inline fn f(x: T) T {
                return @abs(x);
            }
        }.f;

        const rot = struct {
            inline fn compute(a_aa: T, a_bb: T, a_ab: T) struct { c: T, s: T } {
                // Diagonalize [[a_aa, a_ab],[a_ab, a_bb]] with a Jacobi rotation.
                if (a_ab == @as(T, 0)) return .{ .c = @as(T, 1), .s = @as(T, 0) };
                const tau: T = (a_bb - a_aa) / (@as(T, 2) * a_ab);
                const sign_tau: T = if (tau >= @as(T, 0)) @as(T, 1) else @as(T, -1);
                const t: T = sign_tau / (absT(tau) + @sqrt(@as(T, 1) + tau * tau));
                const c: T = @as(T, 1) / @sqrt(@as(T, 1) + t * t);
                return .{ .c = c, .s = c * t };
            }
        };

        if (m >= n) {
            // Column orthogonalization. Accumulate Vt directly (rows) so we don't need a transpose later.
            // vt := I (n×n)
            for (0..k) |i| {
                const off = i * vt.stride;
                for (0..k) |j| vt.data[off + j] = if (i == j) @as(T, 1) else @as(T, 0);
            }

            var sweep: usize = 0;
            while (sweep < max_sweeps) : (sweep += 1) {
                var changed = false;
                for (0..k) |p| {
                    for (p + 1..k) |q| {
                        // a = ||col_p||^2, b = ||col_q||^2, c = col_p^T col_q
                        var aa: T = @as(T, 0);
                        var bb: T = @as(T, 0);
                        var ab: T = @as(T, 0);
                        for (0..m) |i| {
                            const ap = a.data[i * a.stride + p];
                            const aq = a.data[i * a.stride + q];
                            aa += ap * ap;
                            bb += aq * aq;
                            ab += ap * aq;
                        }

                        if (absT(ab) <= eps * @sqrt(aa * bb)) continue;

                        const r = rot.compute(aa, bb, ab);
                        const c = r.c;
                        const sn = r.s;

                        // A := A * G  (mix columns p,q)
                        for (0..m) |i| {
                            const idx_p = i * a.stride + p;
                            const idx_q = i * a.stride + q;
                            const ap = a.data[idx_p];
                            const aq = a.data[idx_q];
                            a.data[idx_p] = c * ap - sn * aq;
                            a.data[idx_q] = sn * ap + c * aq;
                        }

                        // Vt := G^T * Vt (mix rows p,q)
                        for (0..k) |j| {
                            const idx_p = p * vt.stride + j;
                            const idx_q = q * vt.stride + j;
                            const vp = vt.data[idx_p];
                            const vq = vt.data[idx_q];
                            vt.data[idx_p] = c * vp - sn * vq;
                            vt.data[idx_q] = sn * vp + c * vq;
                        }

                        changed = true;
                    }
                }
                if (!changed) break;
            }
            if (sweep == max_sweeps) return error.NoConvergence;

            // s[j] = ||col_j||, u(:,j) = col_j / s[j]
            for (0..k) |j| {
                var ss: T = @as(T, 0);
                for (0..m) |i| {
                    const v = a.data[i * a.stride + j];
                    ss += v * v;
                }
                const sj: T = @sqrt(@max(ss, @as(T, 0)));
                s[j] = sj;

                if (sj == @as(T, 0)) {
                    for (0..m) |i| u.data[i * u.stride + j] = @as(T, 0);
                } else {
                    const inv = @as(T, 1) / sj;
                    for (0..m) |i| u.data[i * u.stride + j] = a.data[i * a.stride + j] * inv;
                }
            }

            // Sort descending by singular value: swap columns of U and rows of Vt.
            for (0..k) |i| {
                var best: usize = i;
                var best_v: T = s[i];
                for (i + 1..k) |j| {
                    if (s[j] > best_v) {
                        best_v = s[j];
                        best = j;
                    }
                }
                if (best == i) continue;

                // swap s
                const tmp_s = s[i];
                s[i] = s[best];
                s[best] = tmp_s;

                // swap columns i,best of U
                for (0..m) |r| {
                    const idx_i = r * u.stride + i;
                    const idx_b = r * u.stride + best;
                    const tmp = u.data[idx_i];
                    u.data[idx_i] = u.data[idx_b];
                    u.data[idx_b] = tmp;
                }

                // swap rows i,best of Vt
                const row_i = i * vt.stride;
                const row_b = best * vt.stride;
                for (0..k) |cidx| {
                    const tmp = vt.data[row_i + cidx];
                    vt.data[row_i + cidx] = vt.data[row_b + cidx];
                    vt.data[row_b + cidx] = tmp;
                }
            }
        } else {
            // Row orthogonalization. Accumulate U (square, m×m), and emit Vt as normalized rows (m×n).
            // U := I (m×m)
            for (0..k) |i| {
                const off = i * u.stride;
                for (0..k) |j| u.data[off + j] = if (i == j) @as(T, 1) else @as(T, 0);
            }

            var sweep: usize = 0;
            while (sweep < max_sweeps) : (sweep += 1) {
                var changed = false;
                for (0..k) |p| {
                    for (p + 1..k) |q| {
                        // aa = ||row_p||^2, bb = ||row_q||^2, ab = row_p * row_q^T
                        var aa: T = @as(T, 0);
                        var bb: T = @as(T, 0);
                        var ab: T = @as(T, 0);
                        const off_p = p * a.stride;
                        const off_q = q * a.stride;
                        for (0..n) |j| {
                            const ap = a.data[off_p + j];
                            const aq = a.data[off_q + j];
                            aa += ap * ap;
                            bb += aq * aq;
                            ab += ap * aq;
                        }

                        if (absT(ab) <= eps * @sqrt(aa * bb)) continue;

                        const r = rot.compute(aa, bb, ab);
                        const c = r.c;
                        const sn = r.s;

                        // A := R^T * A  (mix rows p,q)
                        for (0..n) |j| {
                            const idx_p = off_p + j;
                            const idx_q = off_q + j;
                            const ap = a.data[idx_p];
                            const aq = a.data[idx_q];
                            a.data[idx_p] = c * ap - sn * aq;
                            a.data[idx_q] = sn * ap + c * aq;
                        }

                        // U := U * R (mix columns p,q)
                        for (0..k) |i| {
                            const idx_p = i * u.stride + p;
                            const idx_q = i * u.stride + q;
                            const up = u.data[idx_p];
                            const uq = u.data[idx_q];
                            u.data[idx_p] = c * up - sn * uq;
                            u.data[idx_q] = sn * up + c * uq;
                        }

                        changed = true;
                    }
                }
                if (!changed) break;
            }
            if (sweep == max_sweeps) return error.NoConvergence;

            // s[i] = ||row_i||, vt[i,:] = row_i / s[i]
            for (0..k) |i| {
                const off = i * a.stride;
                var ss: T = @as(T, 0);
                for (0..n) |j| {
                    const v = a.data[off + j];
                    ss += v * v;
                }
                const si: T = @sqrt(@max(ss, @as(T, 0)));
                s[i] = si;

                const out_off = i * vt.stride;
                if (si == @as(T, 0)) {
                    for (0..n) |j| vt.data[out_off + j] = @as(T, 0);
                } else {
                    const inv = @as(T, 1) / si;
                    for (0..n) |j| vt.data[out_off + j] = a.data[off + j] * inv;
                }
            }

            // Sort descending: swap columns of U and rows of Vt.
            for (0..k) |i| {
                var best: usize = i;
                var best_v: T = s[i];
                for (i + 1..k) |j| {
                    if (s[j] > best_v) {
                        best_v = s[j];
                        best = j;
                    }
                }
                if (best == i) continue;

                const tmp_s = s[i];
                s[i] = s[best];
                s[best] = tmp_s;

                // swap columns in U (m×m)
                for (0..k) |r| {
                    const idx_i = r * u.stride + i;
                    const idx_b = r * u.stride + best;
                    const tmp = u.data[idx_i];
                    u.data[idx_i] = u.data[idx_b];
                    u.data[idx_b] = tmp;
                }

                // swap rows in Vt (m×n)
                const row_i = i * vt.stride;
                const row_b = best * vt.stride;
                for (0..n) |cidx| {
                    const tmp = vt.data[row_i + cidx];
                    vt.data[row_i + cidx] = vt.data[row_b + cidx];
                    vt.data[row_b + cidx] = tmp;
                }
            }
        }
    }

    /// Eigen decomposition for a real **symmetric** matrix: `A = V * diag(w) * V^T`.
    ///
    /// - `a` is overwritten (used as workspace).
    /// - `w` length >= `n` receives eigenvalues (descending).
    /// - `v` must be `n×n` and receives eigenvectors as **columns**.
    ///
    /// Current implementation uses a Jacobi rotation method.
    pub fn eig(
        comptime T: type,
        a: *matrix.Matrix(T, .row_major),
        w: []T,
        v: *matrix.Matrix(T, .row_major),
    ) errors.EigError!void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.eig currently only supports floating-point types");
        }

        const n: usize = a.rows;
        std.debug.assert(a.cols == n);
        std.debug.assert(a.stride >= n);
        std.debug.assert(w.len >= n);
        std.debug.assert(v.rows == n and v.cols == n);
        std.debug.assert(v.stride >= n);
        if (n == 0) return;

        // V := I
        for (0..n) |i| {
            const off = i * v.stride;
            for (0..n) |j| v.data[off + j] = if (i == j) @as(T, 1) else @as(T, 0);
        }

        const eps: T = @sqrt(std.math.floatEps(T));
        const max_sweeps: usize = 64 * n;

        var sweep: usize = 0;
        while (sweep < max_sweeps) : (sweep += 1) {
            // Find largest off-diagonal element.
            var p: usize = 0;
            var q: usize = 1;
            var max_ab: T = @as(T, 0);
            for (0..n) |i| {
                for (i + 1..n) |j| {
                    const ab = @abs(a.data[i * a.stride + j]);
                    if (ab > max_ab) {
                        max_ab = ab;
                        p = i;
                        q = j;
                    }
                }
            }

            if (max_ab == @as(T, 0)) break;

            const app = a.data[p * a.stride + p];
            const aqq = a.data[q * a.stride + q];
            const apq = a.data[p * a.stride + q];

            if (@abs(apq) <= eps * (@abs(app) + @abs(aqq))) break;

            // Jacobi rotation to zero apq.
            const tau: T = (aqq - app) / (@as(T, 2) * apq);
            const sign_tau: T = if (tau >= @as(T, 0)) @as(T, 1) else @as(T, -1);
            const t: T = sign_tau / (@abs(tau) + @sqrt(@as(T, 1) + tau * tau));
            const c: T = @as(T, 1) / @sqrt(@as(T, 1) + t * t);
            const s_: T = c * t;

            // Update A = J^T A J (symmetric).
            const app_new: T = c * c * app - @as(T, 2) * c * s_ * apq + s_ * s_ * aqq;
            const aqq_new: T = s_ * s_ * app + @as(T, 2) * c * s_ * apq + c * c * aqq;
            a.data[p * a.stride + p] = app_new;
            a.data[q * a.stride + q] = aqq_new;
            a.data[p * a.stride + q] = @as(T, 0);
            a.data[q * a.stride + p] = @as(T, 0);

            for (0..n) |i| {
                if (i == p or i == q) continue;
                const aip = a.data[i * a.stride + p];
                const aiq = a.data[i * a.stride + q];
                const new_ip: T = c * aip - s_ * aiq;
                const new_iq: T = s_ * aip + c * aiq;
                a.data[i * a.stride + p] = new_ip;
                a.data[p * a.stride + i] = new_ip;
                a.data[i * a.stride + q] = new_iq;
                a.data[q * a.stride + i] = new_iq;
            }

            // Update eigenvectors: V := V * J
            for (0..n) |i| {
                const idx_p = i * v.stride + p;
                const idx_q = i * v.stride + q;
                const vip = v.data[idx_p];
                const viq = v.data[idx_q];
                v.data[idx_p] = c * vip - s_ * viq;
                v.data[idx_q] = s_ * vip + c * viq;
            }
        }
        if (sweep == max_sweeps) return error.NoConvergence;

        // Eigenvalues are diagonal.
        for (0..n) |i| w[i] = a.data[i * a.stride + i];

        // Sort descending by eigenvalue; swap columns of V accordingly.
        for (0..n) |i| {
            var best: usize = i;
            var best_v: T = w[i];
            for (i + 1..n) |j| {
                if (w[j] > best_v) {
                    best_v = w[j];
                    best = j;
                }
            }
            if (best == i) continue;

            const tmpw = w[i];
            w[i] = w[best];
            w[best] = tmpw;

            for (0..n) |r| {
                const idx_i = r * v.stride + i;
                const idx_b = r * v.stride + best;
                const tmp = v.data[idx_i];
                v.data[idx_i] = v.data[idx_b];
                v.data[idx_b] = tmp;
            }
        }
    }

    /// Index of the element with maximum absolute value.
    ///
    /// - **Tie-break**: first occurrence wins (lowest index).
    /// - **NaN policy (floats)**: if any element is NaN, return the index of the *first* NaN.
    /// - **Empty**: returns `null`.
    pub fn iamax(comptime T: type, x: []const T) ?usize {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.iamax currently only supports floating-point types");
        }
        if (x.len == 0) return null;

        // NaN policy: first NaN wins.
        if (std.math.isNan(x[0])) return 0;

        var best_i: usize = 0;
        var best_abs: T = @abs(x[0]);

        if (simd.suggestVectorLength(T)) |vl_c| {
            const VL: usize = @intCast(vl_c);
            if (VL > 1 and x.len >= VL) {
                const Vec = @Vector(VL, T);

                var i: usize = 0;
                const vec_end = x.len - (x.len % VL);
                while (i < vec_end) : (i += VL) {
                    const vx: Vec = x[i..][0..VL].*;
                    const abs_v: Vec = @abs(vx);
                    const abs_arr: [VL]T = @bitCast(abs_v);

                    inline for (0..VL) |lane| {
                        const idx = i + lane;
                        const v = abs_arr[lane];
                        if (std.math.isNan(v)) return idx;
                        if (v > best_abs) {
                            best_abs = v;
                            best_i = idx;
                        }
                    }
                }

                while (i < x.len) : (i += 1) {
                    const v = @abs(x[i]);
                    if (std.math.isNan(v)) return i;
                    if (v > best_abs) {
                        best_abs = v;
                        best_i = i;
                    }
                }
                return best_i;
            }
        }

        for (1..x.len) |i| {
            const v = @abs(x[i]);
            if (std.math.isNan(v)) return i;
            if (v > best_abs) {
                best_abs = v;
                best_i = i;
            }
        }
        return best_i;
    }

    /// Index of the element with minimum absolute value.
    ///
    /// - **Tie-break**: first occurrence wins (lowest index).
    /// - **NaN policy (floats)**: if any element is NaN, return the index of the *first* NaN.
    /// - **Empty**: returns `null`.
    pub fn iamin(comptime T: type, x: []const T) ?usize {
        if (comptime @typeInfo(T) != .float) {
            @compileError("ops.iamin currently only supports floating-point types");
        }
        if (x.len == 0) return null;

        if (std.math.isNan(x[0])) return 0;

        var best_i: usize = 0;
        var best_abs: T = @abs(x[0]);

        if (simd.suggestVectorLength(T)) |vl_c| {
            const VL: usize = @intCast(vl_c);
            if (VL > 1 and x.len >= VL) {
                const Vec = @Vector(VL, T);

                var i: usize = 0;
                const vec_end = x.len - (x.len % VL);
                while (i < vec_end) : (i += VL) {
                    const vx: Vec = x[i..][0..VL].*;
                    const abs_v: Vec = @abs(vx);
                    const abs_arr: [VL]T = @bitCast(abs_v);

                    inline for (0..VL) |lane| {
                        const idx = i + lane;
                        const v = abs_arr[lane];
                        if (std.math.isNan(v)) return idx;
                        if (v < best_abs) {
                            best_abs = v;
                            best_i = idx;
                        }
                    }
                }

                while (i < x.len) : (i += 1) {
                    const v = @abs(x[i]);
                    if (std.math.isNan(v)) return i;
                    if (v < best_abs) {
                        best_abs = v;
                        best_i = i;
                    }
                }
                return best_i;
            }
        }

        for (1..x.len) |i| {
            const v = @abs(x[i]);
            if (std.math.isNan(v)) return i;
            if (v < best_abs) {
                best_abs = v;
                best_i = i;
            }
        }
        return best_i;
    }
};


