const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn fillTriangularPoisonOther(
    comptime T: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    diag: blazt.Diag,
    a: *blazt.Matrix(T, layout),
) void {
    std.debug.assert(a.rows == a.cols);
    const n = a.rows;

    for (0..n) |j| {
        for (0..n) |i| {
            const in_stored = switch (uplo) {
                .upper => i <= j,
                .lower => i >= j,
            };
            if (!in_stored) {
                a.atPtr(i, j).* = std.math.nan(T);
                continue;
            }

            if (i == j) {
                a.atPtr(i, j).* = if (diag == .unit)
                    std.math.nan(T) // must not be read
                else
                    @as(T, 1.5) + @as(T, @floatFromInt(@as(u32, @intCast((i + 1) % 7)))) * @as(T, 0.01);
            } else {
                const s: u32 = @intCast((i * 37 + j * 17 + 11) % 1024);
                a.atPtr(i, j).* = @as(T, @floatFromInt(s)) * @as(T, 0.001);
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

fn triOpAt(
    comptime T: type,
    comptime layout: blazt.Layout,
    a: blazt.Matrix(T, layout),
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    diag: blazt.Diag,
    i: usize,
    j: usize,
) T {
    const eff_trans: blazt.Trans = switch (trans) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };
    const op_uplo: blazt.UpLo = if (eff_trans == .no_trans)
        uplo
    else
        switch (uplo) {
            .upper => .lower,
            .lower => .upper,
        };

    const aAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), ii: usize, jj: usize) T {
            return switch (layout) {
                .row_major => mat.data[ii * mat.stride + jj],
                .col_major => mat.data[jj * mat.stride + ii],
            };
        }
    }.f;
    const aOp = struct {
        inline fn f(mat: blazt.Matrix(T, layout), ii: usize, jj: usize, tr: blazt.Trans) T {
            return if (tr == .no_trans) aAt(mat, ii, jj) else aAt(mat, jj, ii);
        }
    }.f;

    const in_tri = switch (op_uplo) {
        .upper => i <= j,
        .lower => i >= j,
    };
    if (!in_tri) return @as(T, 0);
    if (i == j and diag == .unit) return @as(T, 1);
    return aOp(a, i, j, eff_trans);
}

fn refTrmm(
    comptime T: type,
    comptime layout: blazt.Layout,
    side: blazt.Side,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    diag: blazt.Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: blazt.Matrix(T, layout),
    b_in: blazt.Matrix(T, layout),
    b_out: *blazt.Matrix(T, layout),
) void {
    std.debug.assert(b_in.rows == m and b_in.cols == n);
    std.debug.assert(b_out.rows == m and b_out.cols == n);

    const bAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    if (alpha == @as(T, 0)) {
        for (0..m) |i| {
            for (0..n) |j| {
                b_out.atPtr(i, j).* = @as(T, 0);
            }
        }
        return;
    }

    switch (side) {
        .left => {
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: T = 0;
                    for (0..m) |k| {
                        const aik = triOpAt(T, layout, a, uplo, trans, diag, i, k);
                        if (aik != @as(T, 0)) sum += aik * bAt(b_in, k, j);
                    }
                    b_out.atPtr(i, j).* = alpha * sum;
                }
            }
        },
        .right => {
            for (0..n) |j| {
                for (0..m) |i| {
                    var sum: T = 0;
                    for (0..n) |k| {
                        const akj = triOpAt(T, layout, a, uplo, trans, diag, k, j);
                        if (akj != @as(T, 0)) sum += bAt(b_in, i, k) * akj;
                    }
                    b_out.atPtr(i, j).* = alpha * sum;
                }
            }
        },
    }
}

test "ops.trmm matches reference and does not read ignored triangle or unit diagonal" {
    const T = f32;
    const alpha: T = 0.75;
    const tol = fx.FloatTolerance{ .abs = 2e-5, .rel = 2e-5, .ulps = 512 };

    const m: usize = 11;
    const n: usize = 7;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.Side.left, blazt.Side.right }) |side| {
            inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
                inline for (.{ blazt.Trans.no_trans, blazt.Trans.trans, blazt.Trans.conj_trans }) |tr| {
                    inline for (.{ blazt.Diag.unit, blazt.Diag.non_unit }) |diag| {
                        const a_dim: usize = if (side == .left) m else n;

                        var a = try blazt.Matrix(T, layout).init(std.testing.allocator, a_dim, a_dim);
                        defer a.deinit();
                        var b = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                        defer b.deinit();
                        var b0 = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                        defer b0.deinit();
                        var bref = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                        defer bref.deinit();

                        fillTriangularPoisonOther(T, layout, uplo, diag, &a);
                        fillMatDeterministic(T, layout, &b);
                        @memcpy(b0.data, b.data);

                        refTrmm(T, layout, side, uplo, tr, diag, m, n, alpha, a, b0, &bref);
                        blazt.ops.trmm(T, layout, side, uplo, tr, diag, m, n, alpha, a, &b);

                        for (0..m) |i| {
                            for (0..n) |j| {
                                const got = b.at(i, j);
                                try std.testing.expect(!std.math.isNan(got));
                                try fx.expectFloatApproxEq(T, bref.at(i, j), got, tol);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn mulTriLeft(
    comptime T: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    diag: blazt.Diag,
    a: blazt.Matrix(T, layout),
    x: blazt.Matrix(T, layout), // m×n
    out: *blazt.Matrix(T, layout),
) void {
    const m = x.rows;
    const n = x.cols;
    std.debug.assert(out.rows == m and out.cols == n);
    const xAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;
    for (0..n) |j| {
        for (0..m) |i| {
            var sum: T = 0;
            for (0..m) |k| {
                const aik = triOpAt(T, layout, a, uplo, trans, diag, i, k);
                if (aik != @as(T, 0)) sum += aik * xAt(x, k, j);
            }
            out.atPtr(i, j).* = sum;
        }
    }
}

fn mulTriRight(
    comptime T: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    diag: blazt.Diag,
    x: blazt.Matrix(T, layout), // m×n
    a: blazt.Matrix(T, layout),
    out: *blazt.Matrix(T, layout),
) void {
    const m = x.rows;
    const n = x.cols;
    std.debug.assert(out.rows == m and out.cols == n);
    const xAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;
    for (0..n) |j| {
        for (0..m) |i| {
            var sum: T = 0;
            for (0..n) |k| {
                const akj = triOpAt(T, layout, a, uplo, trans, diag, k, j);
                if (akj != @as(T, 0)) sum += xAt(x, i, k) * akj;
            }
            out.atPtr(i, j).* = sum;
        }
    }
}

test "ops.trsm solves correctly (residual check) and does not read ignored triangle or unit diagonal" {
    const T = f32;
    const alpha: T = 0.5;
    const tol = fx.FloatTolerance{ .abs = 3e-4, .rel = 3e-4, .ulps = 2048 };

    const m: usize = 9;
    const n: usize = 7;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.Side.left, blazt.Side.right }) |side| {
            inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
                inline for (.{ blazt.Trans.no_trans, blazt.Trans.trans, blazt.Trans.conj_trans }) |tr| {
                    inline for (.{ blazt.Diag.unit, blazt.Diag.non_unit }) |diag| {
                        const a_dim: usize = if (side == .left) m else n;

                        var a = try blazt.Matrix(T, layout).init(std.testing.allocator, a_dim, a_dim);
                        defer a.deinit();
                        var b = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                        defer b.deinit();
                        var b0 = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                        defer b0.deinit();
                        var lhs = try blazt.Matrix(T, layout).init(std.testing.allocator, m, n);
                        defer lhs.deinit();

                        fillTriangularPoisonOther(T, layout, uplo, diag, &a);
                        fillMatDeterministic(T, layout, &b);
                        @memcpy(b0.data, b.data);

                        try blazt.ops.trsm(T, layout, side, uplo, tr, diag, m, n, alpha, a, &b);

                        // Residual check: side-left => op(A)*X == alpha*B0; side-right => X*op(A) == alpha*B0.
                        switch (side) {
                            .left => mulTriLeft(T, layout, uplo, tr, diag, a, b, &lhs),
                            .right => mulTriRight(T, layout, uplo, tr, diag, b, a, &lhs),
                        }

                        for (0..m) |i| {
                            for (0..n) |j| {
                                const got = lhs.at(i, j);
                                try std.testing.expect(!std.math.isNan(got));
                                try fx.expectFloatApproxEq(T, alpha * b0.at(i, j), got, tol);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "ops.trsm returns error.Singular for non-unit diagonal with a zero diagonal entry (alpha!=0), but alpha==0 short-circuits" {
    const T = f32;
    const m: usize = 5;
    const n: usize = 4;

    var a = try blazt.Matrix(T, .col_major).init(std.testing.allocator, m, m);
    defer a.deinit();
    var b = try blazt.Matrix(T, .col_major).init(std.testing.allocator, m, n);
    defer b.deinit();
    fillTriangularPoisonOther(T, .col_major, .upper, .non_unit, &a);
    fillMatDeterministic(T, .col_major, &b);

    // Force singular diagonal.
    a.atPtr(2, 2).* = 0;

    try std.testing.expectError(error.Singular, blazt.ops.trsm(T, .col_major, .left, .upper, .no_trans, .non_unit, m, n, 1.0, a, &b));

    // alpha==0: should zero B and return without checking singular.
    fillMatDeterministic(T, .col_major, &b);
    try blazt.ops.trsm(T, .col_major, .left, .upper, .no_trans, .non_unit, m, n, 0.0, a, &b);
    for (0..m) |i| for (0..n) |j| try std.testing.expectEqual(@as(T, 0), b.at(i, j));
}


