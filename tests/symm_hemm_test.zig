const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn fillTriangularSymmPoisonOther(
    comptime T: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    a: *blazt.Matrix(T, layout),
) void {
    std.debug.assert(a.rows == a.cols);
    const n = a.rows;

    // Fill stored triangle with deterministic finite values, and poison the other triangle with NaNs.
    for (0..n) |j| {
        for (0..n) |i| {
            const in_stored = switch (uplo) {
                .upper => i <= j,
                .lower => i >= j,
            };
            if (in_stored) {
                const s: u32 = @intCast((i * 37 + j * 17 + 11) % 1024);
                a.atPtr(i, j).* = @as(T, @floatFromInt(s)) * @as(T, 0.001);
            } else {
                a.atPtr(i, j).* = std.math.nan(T);
            }
        }
    }
}

fn fillMatDeterministic(
    comptime T: type,
    comptime layout: blazt.Layout,
    m: *blazt.Matrix(T, layout),
) void {
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            const s: u32 = @intCast((i * 131 + j * 19 + 3) % 1024);
            m.atPtr(i, j).* = @as(T, @floatFromInt(s)) * @as(T, 0.002);
        }
    }
}

fn refSymm(
    comptime T: type,
    comptime layout: blazt.Layout,
    side: blazt.Side,
    uplo: blazt.UpLo,
    m: usize,
    n: usize,
    alpha: T,
    a: blazt.Matrix(T, layout),
    b: blazt.Matrix(T, layout),
    beta: T,
    c0: blazt.Matrix(T, layout),
    c_out: *blazt.Matrix(T, layout),
) void {
    std.debug.assert(c0.rows == m and c0.cols == n);
    std.debug.assert(c_out.rows == m and c_out.cols == n);

    // Copy + scale C by beta.
    @memcpy(c_out.data, c0.data);
    if (beta == @as(T, 0)) {
        @memset(c_out.data, @as(T, 0));
    } else if (beta != @as(T, 1)) {
        for (c_out.data) |*v| v.* *= beta;
    }

    if (alpha == @as(T, 0) or m == 0 or n == 0) return;

    const aAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;
    const bAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    const symmAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), ul: blazt.UpLo, i: usize, j: usize) T {
            if (i == j) return aAt(mat, i, j);
            return switch (ul) {
                .upper => if (i < j) aAt(mat, i, j) else aAt(mat, j, i),
                .lower => if (i > j) aAt(mat, i, j) else aAt(mat, j, i),
            };
        }
    }.f;

    switch (side) {
        .left => {
            // C := alpha*A*B + C
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: T = 0;
                    for (0..m) |p| sum += symmAt(a, uplo, i, p) * bAt(b, p, j);
                    c_out.atPtr(i, j).* += alpha * sum;
                }
            }
        },
        .right => {
            // C := alpha*B*A + C
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: T = 0;
                    for (0..n) |p| sum += bAt(b, i, p) * symmAt(a, uplo, p, j);
                    c_out.atPtr(i, j).* += alpha * sum;
                }
            }
        },
    }
}

test "ops.symm matches reference and does not read the ignored triangle (poisoned with NaN)" {
    const T = f32;
    const alpha: T = 0.75;
    const beta: T = 0.25;
    const tol = fx.FloatTolerance{ .abs = 2e-5, .rel = 2e-5, .ulps = 512 };

    const m: usize = 13;
    const n: usize = 9;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.Side.left, blazt.Side.right }) |side| {
            inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
                const a_dim: usize = if (side == .left) m else n;

                var a = try blazt.Matrix(T, layout).init(std.testing.allocator, a_dim, a_dim);
                defer a.deinit();
                var b = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                defer b.deinit();
                var c = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                defer c.deinit();
                var c0 = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                defer c0.deinit();
                var cref = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                defer cref.deinit();

                fillTriangularSymmPoisonOther(T, layout, uplo, &a);
                fillMatDeterministic(T, layout, &b);
                fillMatDeterministic(T, layout, &c);
                @memcpy(c0.data, c.data);

                blazt.ops.symm(T, layout, side, uplo, m, n, alpha, a, b, beta, &c);
                refSymm(T, layout, side, uplo, m, n, alpha, a, b, beta, c0, &cref);

                for (0..m) |i| {
                    for (0..n) |j| {
                        const got = c.at(i, j);
                        try std.testing.expect(!std.math.isNan(got));
                        try fx.expectFloatApproxEq(T, cref.at(i, j), got, tol);
                    }
                }
            }
        }
    }
}

fn fillTriangularHermPoisonOther(
    comptime C: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    a: *blazt.Matrix(C, layout),
) void {
    std.debug.assert(a.rows == a.cols);
    const n = a.rows;
    const R = @TypeOf(@as(C, undefined).re);
    const nan_r: R = std.math.nan(R);

    for (0..n) |j| {
        for (0..n) |i| {
            const in_stored = switch (uplo) {
                .upper => i <= j,
                .lower => i >= j,
            };
            if (in_stored) {
                const re: R = @as(R, @floatFromInt(@as(u32, @intCast((i * 31 + j * 7 + 1) % 251)))) * @as(R, 0.01);
                const im: R = if (i == j) @as(R, 0) else @as(R, @floatFromInt(@as(u32, @intCast((i * 13 + j * 11 + 3) % 251)))) * @as(R, 0.01);
                a.atPtr(i, j).* = C.init(re, im);
            } else {
                a.atPtr(i, j).* = C.init(nan_r, nan_r);
            }
        }
    }
}

fn fillMatDeterministicComplex(
    comptime C: type,
    comptime layout: blazt.Layout,
    m: *blazt.Matrix(C, layout),
) void {
    const R = @TypeOf(@as(C, undefined).re);
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            const re: R = @as(R, @floatFromInt(@as(u32, @intCast((i * 19 + j * 23 + 7) % 257)))) * @as(R, 0.01);
            const im: R = @as(R, @floatFromInt(@as(u32, @intCast((i * 11 + j * 29 + 3) % 257)))) * @as(R, 0.01);
            m.atPtr(i, j).* = C.init(re, im);
        }
    }
}

fn refHemm(
    comptime C: type,
    comptime layout: blazt.Layout,
    side: blazt.Side,
    uplo: blazt.UpLo,
    m: usize,
    n: usize,
    alpha: C,
    a: blazt.Matrix(C, layout),
    b: blazt.Matrix(C, layout),
    beta: C,
    c0: blazt.Matrix(C, layout),
    c_out: *blazt.Matrix(C, layout),
) void {
    const R = @TypeOf(@as(C, undefined).re);
    const zero: C = C.init(@as(R, 0), @as(R, 0));

    @memcpy(c_out.data, c0.data);
    const beta_is_zero = (beta.re == @as(R, 0)) and (beta.im == @as(R, 0));
    const beta_is_one = (beta.re == @as(R, 1)) and (beta.im == @as(R, 0));
    if (beta_is_zero) {
        for (c_out.data) |*v| v.* = zero;
    } else if (!beta_is_one) {
        for (c_out.data) |*v| v.* = v.*.mul(beta);
    }

    const alpha_is_zero = (alpha.re == @as(R, 0)) and (alpha.im == @as(R, 0));
    if (alpha_is_zero or m == 0 or n == 0) return;

    const aAt = struct {
        inline fn f(mat: blazt.Matrix(C, layout), i: usize, j: usize) C {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;
    const bAt = struct {
        inline fn f(mat: blazt.Matrix(C, layout), i: usize, j: usize) C {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    const hermAt = struct {
        inline fn f(mat: blazt.Matrix(C, layout), ul: blazt.UpLo, i: usize, j: usize) C {
            const diagReal = struct {
                inline fn g(v: C) C {
                    return C.init(v.re, @as(R, 0));
                }
            }.g;
            if (i == j) return diagReal(aAt(mat, i, j));
            return switch (ul) {
                .upper => if (i < j) aAt(mat, i, j) else aAt(mat, j, i).conjugate(),
                .lower => if (i > j) aAt(mat, i, j) else aAt(mat, j, i).conjugate(),
            };
        }
    }.f;

    switch (side) {
        .left => {
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: C = zero;
                    for (0..m) |p| sum = sum.add(hermAt(a, uplo, i, p).mul(bAt(b, p, j)));
                    c_out.atPtr(i, j).* = c_out.at(i, j).add(alpha.mul(sum));
                }
            }
        },
        .right => {
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: C = zero;
                    for (0..n) |p| sum = sum.add(bAt(b, i, p).mul(hermAt(a, uplo, p, j)));
                    c_out.atPtr(i, j).* = c_out.at(i, j).add(alpha.mul(sum));
                }
            }
        },
    }
}

test "ops.hemm matches reference and does not read the ignored triangle (poisoned with NaN)" {
    const C = std.math.Complex(f32);
    const R = f32;
    const alpha: C = C.init(0.5, -0.25);
    const beta: C = C.init(0.25, 0.1);
    const tol = fx.FloatTolerance{ .abs = 3e-4, .rel = 3e-4, .ulps = 2048 };

    const m: usize = 9;
    const n: usize = 7;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.Side.left, blazt.Side.right }) |side| {
            inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
                const a_dim: usize = if (side == .left) m else n;

                var a = try blazt.Matrix(C, layout).init(std.testing.allocator, a_dim, a_dim);
                defer a.deinit();
                var b = try blazt.Matrix(C, layout).init(std.testing.allocator, m, n);
                defer b.deinit();
                var c = try blazt.Matrix(C, layout).init(std.testing.allocator, m, n);
                defer c.deinit();
                var c0 = try blazt.Matrix(C, layout).init(std.testing.allocator, m, n);
                defer c0.deinit();
                var cref = try blazt.Matrix(C, layout).init(std.testing.allocator, m, n);
                defer cref.deinit();

                fillTriangularHermPoisonOther(C, layout, uplo, &a);
                fillMatDeterministicComplex(C, layout, &b);
                fillMatDeterministicComplex(C, layout, &c);
                @memcpy(c0.data, c.data);

                blazt.ops.hemm(C, layout, side, uplo, m, n, alpha, a, b, beta, &c);
                refHemm(C, layout, side, uplo, m, n, alpha, a, b, beta, c0, &cref);

                for (0..m) |i| {
                    for (0..n) |j| {
                        const got = c.at(i, j);
                        try std.testing.expect(!std.math.isNan(got.re));
                        try std.testing.expect(!std.math.isNan(got.im));
                        const exp = cref.at(i, j);
                        try fx.expectFloatApproxEq(R, exp.re, got.re, tol);
                        try fx.expectFloatApproxEq(R, exp.im, got.im, tol);
                    }
                }
            }
        }
    }
}


