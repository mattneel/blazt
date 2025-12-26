const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn tolFor(comptime T: type) fx.FloatTolerance {
    return switch (T) {
        f32 => .{ .abs = 3e-2, .rel = 3e-2, .ulps = 1 << 22 },
        f64 => .{ .abs = 6e-8, .rel = 6e-8, .ulps = 1 << 22 },
        else => @compileError("tolFor only supports f32/f64"),
    };
}

test "ops.eig (symmetric): Av ~= V*diag(w) and reconstruction" {
    inline for (.{ f32, f64 }) |T| {
        const tol = tolFor(T);

        var rng = fx.FixtureRng.init(0x1ee7_1ee7_1ee7_1ee7);
        const r = rng.random();

        const n: usize = 16;
        var m = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, n, n);
        defer m.deinit();

        // Symmetrize: A0 = 0.5*(M + M^T)
        var a0 = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer a0.deinit();
        for (0..n) |i| {
            for (0..n) |j| {
                const v = (m.at(i, j) + m.at(j, i)) * @as(T, 0.5);
                a0.atPtr(i, j).* = v;
            }
        }

        var a = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memcpy(a.data, a0.data);

        var v = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer v.deinit();
        const w = try std.testing.allocator.alloc(T, n);
        defer std.testing.allocator.free(w);

        try blazt.ops.eig(T, &a, w, &v);

        // Eigenvalues sorted.
        for (0..n - 1) |i| try std.testing.expect(w[i] >= w[i + 1]);

        // Orthogonality: V^T V ~= I
        var vtv = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer vtv.deinit();
        @memset(vtv.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .trans, .no_trans, @as(T, 1), v, v, @as(T, 0), &vtv);
        for (0..n) |i| {
            for (0..n) |j| {
                const exp: T = if (i == j) @as(T, 1) else @as(T, 0);
                try fx.expectFloatApproxEq(T, exp, vtv.at(i, j), tol);
            }
        }

        // AV vs V*diag(w)
        var av = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer av.deinit();
        @memset(av.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, @as(T, 1), a0, v, @as(T, 0), &av);
        for (0..n) |i| {
            for (0..n) |j| {
                const rhs = v.at(i, j) * w[j];
                try fx.expectFloatApproxEq(T, rhs, av.at(i, j), tol);
            }
        }

        // Reconstruction: V*diag(w)*V^T
        var vw = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer vw.deinit();
        for (0..n) |i| {
            const off = i * v.stride;
            const out_off = i * vw.stride;
            for (0..n) |j| {
                vw.data[out_off + j] = v.data[off + j] * w[j];
            }
        }

        var recon = try blazt.Matrix(T, .row_major).init(std.testing.allocator, n, n);
        defer recon.deinit();
        @memset(recon.data, @as(T, 0));
        blazt.ops.gemm(T, .row_major, .no_trans, .trans, @as(T, 1), vw, v, @as(T, 0), &recon);

        for (0..n) |i| {
            for (0..n) |j| {
                try fx.expectFloatApproxEq(T, a0.at(i, j), recon.at(i, j), tol);
            }
        }
    }
}


