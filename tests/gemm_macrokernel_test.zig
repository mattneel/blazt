const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn refGemmF64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: blazt.Matrix(f32, .col_major),
    b: blazt.Matrix(f32, .col_major),
    beta: f32,
    c_in: []const f32,
    c_out: []f32,
) void {
    const ldc = a.rows; // for col_major C stride equals m
    _ = ldc;

    for (0..n) |j| {
        for (0..m) |i| {
            var sum: f64 = 0.0;
            for (0..k) |p| {
                sum += @as(f64, @floatCast(a.at(i, p))) * @as(f64, @floatCast(b.at(p, j)));
            }

            const idx = i + j * m;
            const cin: f64 = if (beta == @as(f32, 0)) 0.0 else @as(f64, @floatCast(c_in[idx]));
            const out = @as(f64, @floatCast(alpha)) * sum + @as(f64, @floatCast(beta)) * cin;
            c_out[idx] = @floatCast(out);
        }
    }
}

fn fillMatDeterministic(comptime layout: blazt.Layout, mat: *blazt.Matrix(f32, layout)) void {
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 131 + j * 17 + 1) % 1024)))) * 0.001;
            mat.atPtr(i, j).* = v;
        }
    }
}

test "gemm.gemmBlocked (col_major, no_trans) matches reference for tail shapes (beta=0 NaN-safe)" {
    const P = blazt.gemm.computeTileParams(f32);

    const pack_a = try blazt.allocAligned(std.testing.allocator, f32, P.MR * P.KC);
    defer std.testing.allocator.free(pack_a);
    const pack_b = try blazt.allocAligned(std.testing.allocator, f32, P.NR * P.KC);
    defer std.testing.allocator.free(pack_b);

    const m: usize = 7;
    const n: usize = 5;
    const k: usize = 9;

    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, k);
    defer a.deinit();
    var b = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, k, n);
    defer b.deinit();
    var c = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer c.deinit();

    fillMatDeterministic(.col_major, &a);
    fillMatDeterministic(.col_major, &b);

    const nan = std.math.nan(f32);
    @memset(c.data, nan);

    const alpha: f32 = 0.5;
    const beta: f32 = 0.0;

    var cref = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer cref.deinit();
    @memset(cref.data, 0);
    refGemmF64(m, n, k, alpha, a, b, beta, c.data, cref.data);

    blazt.gemm.gemmBlocked(f32, m, n, k, alpha, a, b, beta, &c, pack_a, pack_b);

    for (c.data) |v| try std.testing.expect(!std.math.isNan(v));
    try fx.expectSliceApproxEq(f32, cref.data, c.data, fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 });
}

test "gemm.gemmBlocked accumulates into C when beta=1" {
    const P = blazt.gemm.computeTileParams(f32);

    const pack_a = try blazt.allocAligned(std.testing.allocator, f32, P.MR * P.KC);
    defer std.testing.allocator.free(pack_a);
    const pack_b = try blazt.allocAligned(std.testing.allocator, f32, P.NR * P.KC);
    defer std.testing.allocator.free(pack_b);

    const m: usize = 9;
    const n: usize = 9;
    const k: usize = 7;

    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, k);
    defer a.deinit();
    var b = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, k, n);
    defer b.deinit();
    var c = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer c.deinit();

    fillMatDeterministic(.col_major, &a);
    fillMatDeterministic(.col_major, &b);

    for (c.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * 0.002;

    const alpha: f32 = 0.5;
    const beta: f32 = 1.0;

    const c_in = try std.testing.allocator.dupe(f32, c.data);
    defer std.testing.allocator.free(c_in);

    var cref = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer cref.deinit();
    @memcpy(cref.data, c.data);
    refGemmF64(m, n, k, alpha, a, b, beta, c_in, cref.data);

    blazt.gemm.gemmBlocked(f32, m, n, k, alpha, a, b, beta, &c, pack_a, pack_b);

    try fx.expectSliceApproxEq(f32, cref.data, c.data, fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 });
}


