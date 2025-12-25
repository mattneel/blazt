const builtin = @import("builtin");
const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn oracleTranspose(t: blazt.Trans) blazt.oracle.Oracle.CblasTranspose {
    return switch (t) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans, // real GEMM: conj-trans == trans
    };
}

fn toggleTranspose(t: blazt.Trans) blazt.oracle.Oracle.CblasTranspose {
    // When using row_major -> col_major transpose views, we need op(X)^T.
    return switch (t) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };
}

fn oracleGemm(
    oracle: *const blazt.oracle.Oracle,
    comptime T: type,
    comptime layout: blazt.Layout,
    trans_a: blazt.Trans,
    trans_b: blazt.Trans,
    alpha: T,
    a: blazt.Matrix(T, layout),
    b: blazt.Matrix(T, layout),
    beta: T,
    c: *blazt.Matrix(T, layout),
) void {
    // Effective transposes for real types.
    const eff_a: blazt.Trans = switch (trans_a) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };
    const eff_b: blazt.Trans = switch (trans_b) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };

    const m: usize = c.rows;
    const n: usize = c.cols;

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

    const mi: c_int = @intCast(m);
    const ni: c_int = @intCast(n);
    const ki: c_int = @intCast(k);

    switch (layout) {
        .col_major => {
            const ta = oracleTranspose(trans_a);
            const tb = oracleTranspose(trans_b);
            const lda: c_int = @intCast(a.stride);
            const ldb: c_int = @intCast(b.stride);
            const ldc: c_int = @intCast(c.stride);

            if (T == f32) {
                oracle.sgemm(.col_major, ta, tb, mi, ni, ki, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
            } else if (T == f64) {
                oracle.dgemm(.col_major, ta, tb, mi, ni, ki, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
            } else {
                @compileError("oracleGemm only supports f32/f64");
            }
        },
        .row_major => {
            // Row-major GEMM via col-major transpose views:
            // (op(A)*op(B))^T = op(B)^T * op(A)^T
            const a_t: blazt.Matrix(T, .col_major) = .{
                .data = a.data,
                .rows = a.cols,
                .cols = a.rows,
                .stride = a.stride,
                .allocator = a.allocator,
            };
            const b_t: blazt.Matrix(T, .col_major) = .{
                .data = b.data,
                .rows = b.cols,
                .cols = b.rows,
                .stride = b.stride,
                .allocator = b.allocator,
            };
            var c_t: blazt.Matrix(T, .col_major) = .{
                .data = c.data,
                .rows = c.cols,
                .cols = c.rows,
                .stride = c.stride,
                .allocator = c.allocator,
            };

            const ta = toggleTranspose(eff_b); // op(B)^T
            const tb = toggleTranspose(eff_a); // op(A)^T
            const lda: c_int = @intCast(b_t.stride);
            const ldb: c_int = @intCast(a_t.stride);
            const ldc: c_int = @intCast(c_t.stride);

            // Swap m/n for the transposed result (Cᵀ is n×m).
            const mi_t: c_int = @intCast(n);
            const ni_t: c_int = @intCast(m);

            if (T == f32) {
                oracle.sgemm(.col_major, ta, tb, mi_t, ni_t, ki, alpha, b_t.data.ptr, lda, a_t.data.ptr, ldb, beta, c_t.data.ptr, ldc);
            } else if (T == f64) {
                oracle.dgemm(.col_major, ta, tb, mi_t, ni_t, ki, alpha, b_t.data.ptr, lda, a_t.data.ptr, ldb, beta, c_t.data.ptr, ldc);
            } else {
                @compileError("oracleGemm only supports f32/f64");
            }
        },
    }
}

fn expectMatApproxEq(
    comptime T: type,
    comptime layout: blazt.Layout,
    expected: blazt.Matrix(T, layout),
    got: blazt.Matrix(T, layout),
    tol: fx.FloatTolerance,
) !void {
    try std.testing.expectEqual(expected.rows, got.rows);
    try std.testing.expectEqual(expected.cols, got.cols);
    for (0..expected.rows) |i| {
        for (0..expected.cols) |j| {
            try fx.expectFloatApproxEq(T, expected.at(i, j), got.at(i, j), tol);
        }
    }
}

test "GEMM parity vs oracle (OpenBLAS/BLIS when available)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const trans_cases = [_]blazt.Trans{ .no_trans, .trans, .conj_trans };

    inline for (.{ f32, f64 }) |T| {
        const tol = switch (T) {
            f32 => fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 2048 },
            f64 => fx.FloatTolerance{ .abs = 1e-11, .rel = 1e-11, .ulps = 4096 },
            else => unreachable,
        };

        // A few fixed tail-heavy sizes + a small random sweep.
        const fixed = [_]struct { m: usize, n: usize, k: usize }{
            .{ .m = 1, .n = 1, .k = 1 },
            .{ .m = 3, .n = 5, .k = 7 },
            .{ .m = 17, .n = 19, .k = 23 },
            .{ .m = 31, .n = 29, .k = 13 },
        };

        const rand_iters: usize = if (builtin.mode == .Debug) 8 else 64;
        var rng = fx.FixtureRng.init(0x6b1d_79c2_9c45_1aef);

        inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
            // Fixed cases.
            for (fixed) |cs| {
                try runOneCase(&oracle, T, layout, trans_cases[0..], tol, cs.m, cs.n, cs.k, rng.random());
            }
            // Random cases.
            for (0..rand_iters) |_| {
                const m = rng.random().intRangeAtMost(usize, 1, 64);
                const n = rng.random().intRangeAtMost(usize, 1, 64);
                const k = rng.random().intRangeAtMost(usize, 1, 64);
                try runOneCase(&oracle, T, layout, trans_cases[0..], tol, m, n, k, rng.random());
            }
        }
    }
}

fn runOneCase(
    oracle: *const blazt.oracle.Oracle,
    comptime T: type,
    comptime layout: blazt.Layout,
    trans_cases: []const blazt.Trans,
    tol: fx.FloatTolerance,
    m: usize,
    n: usize,
    k: usize,
    r: std.Random,
) !void {
    const alpha: T = fx.randomScalar(r, T);
    const beta0: T = @as(T, 0);
    const beta1: T = @as(T, 1);

    for (trans_cases) |ta| {
        for (trans_cases) |tb| {
            const eff_a: blazt.Trans = switch (ta) {
                .no_trans => .no_trans,
                .trans, .conj_trans => .trans,
            };
            const eff_b: blazt.Trans = switch (tb) {
                .no_trans => .no_trans,
                .trans, .conj_trans => .trans,
            };

            const a_rows: usize = if (eff_a == .no_trans) m else k;
            const a_cols: usize = if (eff_a == .no_trans) k else m;
            const b_rows: usize = if (eff_b == .no_trans) k else n;
            const b_cols: usize = if (eff_b == .no_trans) n else k;

            var a = try fx.randomMatrix(r, T, layout, std.testing.allocator, a_rows, a_cols);
            defer a.deinit();
            var b = try fx.randomMatrix(r, T, layout, std.testing.allocator, b_rows, b_cols);
            defer b.deinit();

            var c0 = try fx.randomMatrix(r, T, layout, std.testing.allocator, m, n);
            defer c0.deinit();

            // beta=0
            var c_blazt0 = try fx.randomMatrix(r, T, layout, std.testing.allocator, m, n);
            defer c_blazt0.deinit();
            var c_oracle0 = try fx.randomMatrix(r, T, layout, std.testing.allocator, m, n);
            defer c_oracle0.deinit();

            // Make initial C identical across impls.
            @memcpy(c_blazt0.data, c0.data);
            @memcpy(c_oracle0.data, c0.data);

            blazt.ops.gemm(T, layout, ta, tb, alpha, a, b, beta0, &c_blazt0);
            oracleGemm(oracle, T, layout, ta, tb, alpha, a, b, beta0, &c_oracle0);
            try expectMatApproxEq(T, layout, c_oracle0, c_blazt0, tol);

            // beta=1
            var c_blazt1 = try fx.randomMatrix(r, T, layout, std.testing.allocator, m, n);
            defer c_blazt1.deinit();
            var c_oracle1 = try fx.randomMatrix(r, T, layout, std.testing.allocator, m, n);
            defer c_oracle1.deinit();

            @memcpy(c_blazt1.data, c0.data);
            @memcpy(c_oracle1.data, c0.data);

            blazt.ops.gemm(T, layout, ta, tb, alpha, a, b, beta1, &c_blazt1);
            oracleGemm(oracle, T, layout, ta, tb, alpha, a, b, beta1, &c_oracle1);
            try expectMatApproxEq(T, layout, c_oracle1, c_blazt1, tol);
        }
    }
}


