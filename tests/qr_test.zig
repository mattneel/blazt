const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn tolFor(comptime T: type) fx.FloatTolerance {
    return switch (T) {
        f32 => .{ .abs = 3e-3, .rel = 3e-3, .ulps = 1 << 21 },
        f64 => .{ .abs = 1e-10, .rel = 1e-10, .ulps = 1 << 21 },
        else => @compileError("tolFor only supports f32/f64"),
    };
}

fn buildR(
    comptime T: type,
    a_qr: blazt.Matrix(T, .row_major),
    r: *blazt.Matrix(T, .row_major),
) void {
    const m = a_qr.rows;
    const n = a_qr.cols;
    std.debug.assert(r.rows == m and r.cols == n);
    for (0..m) |i| {
        for (0..n) |j| {
            r.atPtr(i, j).* = if (i <= j) a_qr.at(i, j) else @as(T, 0);
        }
    }
}

test "ops.qr: reconstruction A == Q*R and orthogonality Q^T*Q ~= I" {
    inline for (.{ f32, f64 }) |T| {
        const tol = tolFor(T);

        var rng = fx.FixtureRng.init(0x1234_5678);
        const r = rng.random();

        const cases = [_][2]usize{
            .{ 32, 16 },
            .{ 16, 32 },
            .{ 32, 32 },
        };

        inline for (cases) |c| {
            const m: usize = c[0];
            const n: usize = c[1];
            const kmax: usize = @min(m, n);

            var a0 = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, m, n);
            defer a0.deinit();
            var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, n);
            defer a.deinit();
            @memcpy(a.data, a0.data);

            const tau = try std.testing.allocator.alloc(T, kmax);
            defer std.testing.allocator.free(tau);

            blazt.ops.qr(T, &a, tau);

            var q = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, m);
            defer q.deinit();
            blazt.ops.qrFormQ(T, a, tau, &q);

            var rmat = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, n);
            defer rmat.deinit();
            buildR(T, a, &rmat);

            var qr = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, n);
            defer qr.deinit();
            @memset(qr.data, @as(T, 0));
            blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, @as(T, 1), q, rmat, @as(T, 0), &qr);

            for (0..m) |i| {
                for (0..n) |j| {
                    try fx.expectFloatApproxEq(T, a0.at(i, j), qr.at(i, j), tol);
                }
            }

            // Orthogonality: Q^T Q == I
            var qtq = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, m);
            defer qtq.deinit();
            @memset(qtq.data, @as(T, 0));
            blazt.ops.gemm(T, .row_major, .trans, .no_trans, @as(T, 1), q, q, @as(T, 0), &qtq);

            for (0..m) |i| {
                for (0..m) |j| {
                    const exp: T = if (i == j) @as(T, 1) else @as(T, 0);
                    try fx.expectFloatApproxEq(T, exp, qtq.at(i, j), tol);
                }
            }
        }
    }
}



