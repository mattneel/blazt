const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn swapRowsRowMajor(comptime T: type, mat: *blazt.Matrix(T, .row_major), r0: usize, r1: usize) void {
    if (r0 == r1) return;
    std.debug.assert(r0 < mat.rows and r1 < mat.rows);
    const n = mat.cols;
    const s = mat.stride;
    const off0 = r0 * s;
    const off1 = r1 * s;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        const tmp = mat.data[off0 + j];
        mat.data[off0 + j] = mat.data[off1 + j];
        mat.data[off1 + j] = tmp;
    }
}

fn buildLU(
    comptime T: type,
    a_lu: blazt.Matrix(T, .row_major),
    l: *blazt.Matrix(T, .row_major),
    u: *blazt.Matrix(T, .row_major),
) void {
    const m = a_lu.rows;
    const n = a_lu.cols;
    std.debug.assert(l.rows == m and l.cols == m);
    std.debug.assert(u.rows == m and u.cols == n);

    // L: unit lower (m×m), U: upper (m×n)
    for (0..m) |i| {
        for (0..m) |j| {
            l.atPtr(i, j).* = if (i == j) @as(T, 1) else if (i > j) a_lu.at(i, j) else @as(T, 0);
        }
    }
    for (0..m) |i| {
        for (0..n) |j| {
            u.atPtr(i, j).* = if (i <= j) a_lu.at(i, j) else @as(T, 0);
        }
    }
}

fn applyPivotsRowMajor(comptime T: type, a: *blazt.Matrix(T, .row_major), ipiv: []const i32) void {
    const min_mn: usize = @min(a.rows, a.cols);
    std.debug.assert(ipiv.len >= min_mn);
    for (0..min_mn) |k| {
        const piv_1: usize = @intCast(ipiv[k]);
        std.debug.assert(piv_1 >= 1 and piv_1 <= a.rows);
        const piv: usize = piv_1 - 1;
        swapRowsRowMajor(T, a, k, piv);
    }
}

fn tolFor(comptime T: type) fx.FloatTolerance {
    return switch (T) {
        f32 => .{ .abs = 2e-3, .rel = 2e-3, .ulps = 1 << 20 },
        f64 => .{ .abs = 1e-10, .rel = 1e-10, .ulps = 1 << 20 },
        else => @compileError("tolFor only supports f32/f64"),
    };
}

test "ops.lu: P*A == L*U (f32/f64)" {
    inline for (.{ f32, f64 }) |T| {
        const tol = tolFor(T);

        var rng = fx.FixtureRng.init(0x1eed_f00d);
        const r = rng.random();

        const n: usize = 32;
        var a0 = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, n, n);
        defer a0.deinit();
        var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memcpy(a.data, a0.data);

        const ipiv = try std.testing.allocator.alloc(i32, n);
        defer std.testing.allocator.free(ipiv);

        try blazt.ops.lu(T, &a, ipiv);

        var pa = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer pa.deinit();
        @memcpy(pa.data, a0.data);
        applyPivotsRowMajor(T, &pa, ipiv);

        var l = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer l.deinit();
        var u = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer u.deinit();
        buildLU(T, a, &l, &u);

        var lu = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer lu.deinit();
        @memset(lu.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, @as(T, 1), l, u, @as(T, 0), &lu);

        for (0..n) |i| {
            for (0..n) |j| {
                try fx.expectFloatApproxEq(T, pa.at(i, j), lu.at(i, j), tol);
            }
        }
    }
}

test "ops.lu: ipiv is 1-based and in range" {
    const T = f32;
    const n: usize = 10;
    var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
    defer a.deinit();
    // Make sure pivoting does something but stays nonsingular.
    for (0..n) |i| {
        for (0..n) |j| {
            a.atPtr(i, j).* = @as(T, @floatFromInt(@as(u32, @intCast((i * 7 + j * 13 + 1) % 97)))) * @as(T, 0.01);
        }
    }
    const ipiv = try std.testing.allocator.alloc(i32, n);
    defer std.testing.allocator.free(ipiv);
    try blazt.ops.lu(T, &a, ipiv);
    for (0..n) |k| {
        try std.testing.expect(ipiv[k] >= 1);
        try std.testing.expect(ipiv[k] <= @as(i32, @intCast(n)));
    }
}

test "ops.lu: returns error.Singular when pivot is zero" {
    const T = f32;
    const n: usize = 8;
    var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
    defer a.deinit();
    @memset(a.data, @as(T, 0));
    const ipiv = try std.testing.allocator.alloc(i32, n);
    defer std.testing.allocator.free(ipiv);

    try std.testing.expectError(error.Singular, blazt.ops.lu(T, &a, ipiv));
}


