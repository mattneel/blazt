const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn tolFor(comptime T: type) fx.FloatTolerance {
    return switch (T) {
        f32 => .{ .abs = 2e-3, .rel = 2e-3, .ulps = 1 << 20 },
        f64 => .{ .abs = 1e-10, .rel = 1e-10, .ulps = 1 << 20 },
        else => @compileError("tolFor only supports f32/f64"),
    };
}

fn buildLowerFactor(comptime T: type, a: blazt.Matrix(T, .row_major), l: *blazt.Matrix(T, .row_major)) void {
    const n = a.rows;
    std.debug.assert(a.cols == n);
    std.debug.assert(l.rows == n and l.cols == n);
    for (0..n) |i| {
        for (0..n) |j| {
            l.atPtr(i, j).* = if (i >= j) a.at(i, j) else @as(T, 0);
        }
    }
}

fn buildUpperFactor(comptime T: type, a: blazt.Matrix(T, .row_major), u: *blazt.Matrix(T, .row_major)) void {
    const n = a.rows;
    std.debug.assert(a.cols == n);
    std.debug.assert(u.rows == n and u.cols == n);
    for (0..n) |i| {
        for (0..n) |j| {
            u.atPtr(i, j).* = if (i <= j) a.at(i, j) else @as(T, 0);
        }
    }
}

test "ops.cholesky reconstructs A (lower/upper) and ignores the other triangle" {
    inline for (.{ f32, f64 }) |T| {
        const tol = tolFor(T);

        var rng = fx.FixtureRng.init(0xdecafbad);
        const r = rng.random();

        const n: usize = 32;
        var m = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, n, n);
        defer m.deinit();

        // A0 := M^T * M + n*I  (SPD)
        var a0 = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer a0.deinit();
        @memset(a0.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .trans, .no_trans, @as(T, 1), m, m, @as(T, 0), &a0);
        for (0..n) |i| a0.atPtr(i, i).* += @as(T, @floatFromInt(@as(u32, @intCast(n))));

        // Lower
        var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memcpy(a.data, a0.data);
        // Poison upper triangle.
        for (0..n) |i| {
            for (i + 1..n) |j| {
                a.atPtr(i, j).* = std.math.nan(T);
            }
        }

        try blazt.ops.cholesky(T, .lower, &a);

        var l = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer l.deinit();
        var recon = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer recon.deinit();
        buildLowerFactor(T, a, &l);
        @memset(recon.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .no_trans, .trans, @as(T, 1), l, l, @as(T, 0), &recon);

        for (0..n) |i| {
            for (0..n) |j| {
                try fx.expectFloatApproxEq(T, a0.at(i, j), recon.at(i, j), tol);
            }
        }

        // Upper
        @memcpy(a.data, a0.data);
        // Poison lower triangle.
        for (0..n) |j| {
            for (j + 1..n) |i| {
                a.atPtr(i, j).* = std.math.nan(T);
            }
        }

        try blazt.ops.cholesky(T, .upper, &a);

        var u = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer u.deinit();
        buildUpperFactor(T, a, &u);
        @memset(recon.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .trans, .no_trans, @as(T, 1), u, u, @as(T, 0), &recon);

        for (0..n) |i| {
            for (0..n) |j| {
                try fx.expectFloatApproxEq(T, a0.at(i, j), recon.at(i, j), tol);
            }
        }
    }
}

test "ops.cholesky rejects non-SPD inputs" {
    const T = f32;
    const n: usize = 8;
    var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
    defer a.deinit();
    @memset(a.data, @as(T, 0));
    // Negative diagonal => not SPD.
    for (0..n) |i| a.atPtr(i, i).* = -1;
    try std.testing.expectError(error.NotPositiveDefinite, blazt.ops.cholesky(T, .lower, &a));
}


