const std = @import("std");
const types = @import("types.zig");
const matrix = @import("matrix.zig");
const errors = @import("errors.zig");
const simd = @import("simd.zig");
const gemm_mod = @import("gemm.zig");
const memory = @import("memory.zig");

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
            std.mem.copyForwards(T, dst, src);
        }
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
                            @memset(c.data[row_off .. row_off + n], @as(T, 0));
                        }
                    },
                    .col_major => {
                        for (0..n) |j| {
                            const col_off = j * c.stride;
                            for (0..m) |i| c.data[col_off + i] = @as(T, 0);
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

    // LAPACK
    pub fn lu(comptime T: type, a: *matrix.Matrix(T, .row_major), ipiv: []i32) errors.LuError!void {
        _ = .{ T, a, ipiv };
        @panic("TODO: ops.lu");
    }

    pub fn cholesky(comptime T: type, uplo: types.UpLo, a: *matrix.Matrix(T, .row_major)) errors.CholeskyError!void {
        _ = .{ T, uplo, a };
        @panic("TODO: ops.cholesky");
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


