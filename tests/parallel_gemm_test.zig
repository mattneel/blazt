const builtin = @import("builtin");
const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn allocStridedRowMajor(
    comptime T: type,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    stride: usize,
) !blazt.Matrix(T, .row_major) {
    std.debug.assert(stride >= cols);
    const count = std.math.mul(usize, rows, stride) catch return error.OutOfMemory;
    const data = try blazt.allocAligned(allocator, T, count);
    return .{
        .data = data,
        .rows = rows,
        .cols = cols,
        .stride = stride,
        .allocator = allocator,
    };
}

fn fillMatDeterministic(comptime T: type, mat: *blazt.Matrix(T, .row_major)) void {
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const s: u32 = @intCast((i * 131 + j * 17 + 1) % 1024);
            const v: T = @as(T, @floatFromInt(s)) * @as(T, 0.001);
            mat.atPtr(i, j).* = v;
        }
    }
}

fn fillMatCInit(comptime T: type, mat: *blazt.Matrix(T, .row_major)) void {
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const s: u32 = @intCast((i * 7 + j * 11 + 3) % 256);
            const v: T = @as(T, @floatFromInt(s)) * @as(T, 0.01);
            mat.atPtr(i, j).* = v;
        }
    }
}

fn expectMatApproxEq(
    comptime T: type,
    expected: blazt.Matrix(T, .row_major),
    got: blazt.Matrix(T, .row_major),
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

test "parallel.gemm matches ops.gemm (row_major, no_trans/no_trans) and is deterministic" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var pool = try blazt.ThreadPool.init(std.testing.allocator, .{
        .thread_count = 4,
        .task_capacity = 4096,
        .deque_capacity = 4096,
    });
    defer pool.deinit();

    const T = f32;
    const tol = fx.FloatTolerance{ .abs = 1e-5, .rel = 1e-5, .ulps = 256 };
    const alpha: T = 0.5;

    const cases = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 1, .n = 1, .k = 1 },
        .{ .m = 17, .n = 19, .k = 23 },
        .{ .m = 64, .n = 48, .k = 33 },
        .{ .m = 123, .n = 97, .k = 55 },
    };

    inline for (cases) |cs| {
        const m = cs.m;
        const n = cs.n;
        const k = cs.k;
        const stride_pad: usize = 7;

        var a = try allocStridedRowMajor(T, std.testing.allocator, m, k, k + stride_pad);
        defer a.deinit();
        var b = try allocStridedRowMajor(T, std.testing.allocator, k, n, n + stride_pad);
        defer b.deinit();

        var c_ref = try allocStridedRowMajor(T, std.testing.allocator, m, n, n + stride_pad);
        defer c_ref.deinit();
        var c_par = try allocStridedRowMajor(T, std.testing.allocator, m, n, n + stride_pad);
        defer c_par.deinit();
        var c_par2 = try allocStridedRowMajor(T, std.testing.allocator, m, n, n + stride_pad);
        defer c_par2.deinit();

        fillMatDeterministic(T, &a);
        fillMatDeterministic(T, &b);

        // beta=0: fill C with NaNs; implementation must not read C.
        const nan = std.math.nan(T);
        @memset(c_ref.data, nan);
        @memset(c_par.data, nan);
        @memset(c_par2.data, nan);

        blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, alpha, a, b, @as(T, 0), &c_ref);
        blazt.parallel.gemm(T, .no_trans, .no_trans, alpha, a, b, @as(T, 0), &c_par, &pool);
        blazt.parallel.gemm(T, .no_trans, .no_trans, alpha, a, b, @as(T, 0), &c_par2, &pool);

        // Determinism: same inputs => identical results.
        for (0..m) |i| {
            for (0..n) |j| {
                try std.testing.expect(!std.math.isNan(c_par.at(i, j)));
                try std.testing.expectEqual(c_par.at(i, j), c_par2.at(i, j));
            }
        }

        try expectMatApproxEq(T, c_ref, c_par, tol);

        // beta=1: accumulates into C.
        fillMatCInit(T, &c_ref);
        fillMatCInit(T, &c_par);

        blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, alpha, a, b, @as(T, 1), &c_ref);
        blazt.parallel.gemm(T, .no_trans, .no_trans, alpha, a, b, @as(T, 1), &c_par, &pool);
        try expectMatApproxEq(T, c_ref, c_par, tol);
    }
}

test "parallel.gemm stress (multiple sizes) matches ops.gemm" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var pool = try blazt.ThreadPool.init(std.testing.allocator, .{
        .thread_count = 4,
        .task_capacity = 4096,
        .deque_capacity = 4096,
    });
    defer pool.deinit();

    var rng = fx.FixtureRng.init(0xdecafbad);
    const r = rng.random();

    const T = f32;
    const tol = fx.FloatTolerance{ .abs = 1e-5, .rel = 1e-5, .ulps = 256 };
    const alpha: T = 0.75;
    const beta: T = 0.25;

    const iters: usize = if (builtin.mode == .Debug) 20 else 200;

    for (0..iters) |_| {
        const m: usize = r.intRangeAtMost(usize, 1, 64);
        const n: usize = r.intRangeAtMost(usize, 1, 64);
        const k: usize = r.intRangeAtMost(usize, 1, 64);

        var a = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, m, k);
        defer a.deinit();
        var b = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, k, n);
        defer b.deinit();

        var c_ref = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, m, n);
        defer c_ref.deinit();
        var c_par = try fx.randomMatrix(r, T, .row_major, std.testing.allocator, m, n);
        defer c_par.deinit();

        // Keep same initial C across both calls.
        @memcpy(c_par.data, c_ref.data);

        blazt.ops.gemm(T, .row_major, .no_trans, .no_trans, alpha, a, b, beta, &c_ref);
        blazt.parallel.gemm(T, .no_trans, .no_trans, alpha, a, b, beta, &c_par, &pool);

        try expectMatApproxEq(T, c_ref, c_par, tol);
    }
}


