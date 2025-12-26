const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn tolFor(comptime T: type) fx.FloatTolerance {
    return switch (T) {
        f32 => .{ .abs = 2e-2, .rel = 2e-2, .ulps = 1 << 22 },
        f64 => .{ .abs = 3e-8, .rel = 3e-8, .ulps = 1 << 22 },
        else => @compileError("tolFor only supports f32/f64"),
    };
}

fn expectOrthoCols(comptime T: type, u: blazt.Matrix(T, .row_major), tol: fx.FloatTolerance) !void {
    const m = u.rows;
    const k = u.cols;
    _ = m;

    var utu = try blazt.Matrix(T, .row_major).init(std.testing.allocator, k, k);
    defer utu.deinit();
    @memset(utu.data, @as(T, 0));
    blazt.ops.gemm(T, .row_major, .trans, .no_trans, @as(T, 1), u, u, @as(T, 0), &utu);

    for (0..k) |i| {
        for (0..k) |j| {
            const exp: T = if (i == j) @as(T, 1) else @as(T, 0);
            try fx.expectFloatApproxEq(T, exp, utu.at(i, j), tol);
        }
    }
}

fn expectOrthoRows(comptime T: type, vt: blazt.Matrix(T, .row_major), tol: fx.FloatTolerance) !void {
    const k = vt.rows;
    const n = vt.cols;
    _ = n;

    var vvt = try blazt.Matrix(T, .row_major).init(std.testing.allocator, k, k);
    defer vvt.deinit();
    @memset(vvt.data, @as(T, 0));
    blazt.ops.gemm(T, .row_major, .no_trans, .trans, @as(T, 1), vt, vt, @as(T, 0), &vvt);

    for (0..k) |i| {
        for (0..k) |j| {
            const exp: T = if (i == j) @as(T, 1) else @as(T, 0);
            try fx.expectFloatApproxEq(T, exp, vvt.at(i, j), tol);
        }
    }
}

test "ops.svd: reconstruction and orthogonality (rectangular)" {
    inline for (.{ f32, f64 }) |T| {
        const tol = tolFor(T);

        var rng = fx.FixtureRng.init(0x5bd5bd5bd5bd5bd5);
        const r = rng.random();

        const cases = [_][2]usize{
            .{ 24, 12 }, // m>n
            .{ 12, 24 }, // m<n
        };

        inline for (cases) |c| {
            const m: usize = c[0];
            const n: usize = c[1];
            const k: usize = @min(m, n);

            var a0 = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, m, n);
            defer a0.deinit();
            var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, n);
            defer a.deinit();
            @memcpy(a.data, a0.data);

            var u = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, k);
            defer u.deinit();
            var vt = try blazt.Matrix(T, .row_major).init(std.testing.allocator, k, n);
            defer vt.deinit();
            const s = try std.testing.allocator.alloc(T, k);
            defer std.testing.allocator.free(s);

            try blazt.ops.svd(T, &a, s, &u, &vt);

            // Singular values sorted and non-negative.
            for (0..k) |i| try std.testing.expect(s[i] >= @as(T, 0));
            for (0..k - 1) |i| try std.testing.expect(s[i] >= s[i + 1]);

            try expectOrthoCols(T, u, tol);
            try expectOrthoRows(T, vt, tol);

            // Recon: U*diag(S)*Vt
            var us = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, k);
            defer us.deinit();
            for (0..m) |i| {
                const off = i * u.stride;
                const out_off = i * us.stride;
                for (0..k) |j| {
                    us.data[out_off + j] = u.data[off + j] * s[j];
                }
            }

            var recon = try blazt.Matrix(T, .row_major).init(std.testing.allocator, m, n);
            defer recon.deinit();
            @memset(recon.data, @as(T, 0));
            blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, @as(T, 1), us, vt, @as(T, 0), &recon);

            for (0..m) |i| {
                for (0..n) |j| {
                    try fx.expectFloatApproxEq(T, a0.at(i, j), recon.at(i, j), tol);
                }
            }
        }
    }
}


