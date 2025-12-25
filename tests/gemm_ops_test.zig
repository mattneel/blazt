const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn allocStridedMatrix(
    comptime T: type,
    comptime layout: blazt.Layout,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    stride: usize,
) !blazt.Matrix(T, layout) {
    const ld_req: usize = switch (layout) {
        .row_major => cols,
        .col_major => rows,
    };
    std.debug.assert(stride >= ld_req);

    const count = switch (layout) {
        .row_major => std.math.mul(usize, rows, stride) catch return error.OutOfMemory,
        .col_major => std.math.mul(usize, cols, stride) catch return error.OutOfMemory,
    };

    const data = try blazt.allocAligned(allocator, T, count);
    return .{
        .data = data,
        .rows = rows,
        .cols = cols,
        .stride = stride,
        .allocator = allocator,
    };
}

fn fillMatDeterministic(comptime T: type, comptime layout: blazt.Layout, mat: *blazt.Matrix(T, layout)) void {
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const s: u32 = @intCast((i * 131 + j * 17 + 1) % 1024);
            const v: T = @as(T, @floatFromInt(s)) * @as(T, 0.001);
            mat.atPtr(i, j).* = v;
        }
    }
}

fn fillMatCInit(comptime T: type, comptime layout: blazt.Layout, mat: *blazt.Matrix(T, layout)) void {
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const s: u32 = @intCast((i * 7 + j * 11 + 3) % 256);
            const v: T = @as(T, @floatFromInt(s)) * @as(T, 0.01);
            mat.atPtr(i, j).* = v;
        }
    }
}

fn computeRefGemm(
    comptime T: type,
    comptime layout: blazt.Layout,
    trans_a: blazt.Trans,
    trans_b: blazt.Trans,
    alpha: T,
    a: blazt.Matrix(T, layout),
    b: blazt.Matrix(T, layout),
    beta: T,
    c_in: blazt.Matrix(T, layout),
    c_out: *blazt.Matrix(T, layout),
) void {
    const eff_a: blazt.Trans = switch (trans_a) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };
    const eff_b: blazt.Trans = switch (trans_b) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };

    const m: usize = c_out.rows;
    const n: usize = c_out.cols;
    std.debug.assert(c_in.rows == m);
    std.debug.assert(c_in.cols == n);

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

    const alpha_f: f64 = @floatCast(alpha);
    const beta_f: f64 = @floatCast(beta);

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0.0;
            for (0..k) |p| {
                const av: T = if (eff_a == .no_trans) aAt(a, i, p) else aAt(a, p, i);
                const bv: T = if (eff_b == .no_trans) bAt(b, p, j) else bAt(b, j, p);
                sum += @as(f64, @floatCast(av)) * @as(f64, @floatCast(bv));
            }

            const cin: f64 = if (beta == @as(T, 0)) 0.0 else @as(f64, @floatCast(c_in.at(i, j)));
            const out: f64 = alpha_f * sum + beta_f * cin;
            c_out.atPtr(i, j).* = @floatCast(out);
        }
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

test "ops.gemm supports layout + trans combos; beta=0 does not read C; stride respected" {
    const T = f32;
    const tol = fx.FloatTolerance{ .abs = 1e-5, .rel = 1e-5, .ulps = 256 };

    const m: usize = 3;
    const n: usize = 2;
    const k: usize = 4;
    const alpha: T = 0.5;
    const stride_pad: usize = 3;

    const trans_cases = [_]blazt.Trans{ .no_trans, .trans, .conj_trans };

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (trans_cases) |ta| {
            inline for (trans_cases) |tb| {
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

                const a_stride: usize = switch (layout) {
                    .row_major => a_cols + stride_pad,
                    .col_major => a_rows + stride_pad,
                };
                const b_stride: usize = switch (layout) {
                    .row_major => b_cols + stride_pad,
                    .col_major => b_rows + stride_pad,
                };
                const c_stride: usize = switch (layout) {
                    .row_major => n + stride_pad,
                    .col_major => m + stride_pad,
                };

                var a = try allocStridedMatrix(T, layout, std.testing.allocator, a_rows, a_cols, a_stride);
                defer a.deinit();
                var b = try allocStridedMatrix(T, layout, std.testing.allocator, b_rows, b_cols, b_stride);
                defer b.deinit();
                var c = try allocStridedMatrix(T, layout, std.testing.allocator, m, n, c_stride);
                defer c.deinit();

                var cref = try allocStridedMatrix(T, layout, std.testing.allocator, m, n, c_stride);
                defer cref.deinit();

                fillMatDeterministic(T, layout, &a);
                fillMatDeterministic(T, layout, &b);

                // beta=0: fill C with NaNs; op must not read C.
                const nan = std.math.nan(T);
                @memset(c.data, nan);
                @memset(cref.data, nan);

                computeRefGemm(T, layout, ta, tb, alpha, a, b, @as(T, 0), cref, &cref);
                blazt.ops.gemm(T, layout, ta, tb, alpha, a, b, @as(T, 0), &c);

                for (0..m) |i| {
                    for (0..n) |j| {
                        try std.testing.expect(!std.math.isNan(c.at(i, j)));
                    }
                }
                try expectMatApproxEq(T, layout, cref, c, tol);

                // beta=1: accumulates into C.
                fillMatCInit(T, layout, &c);
                fillMatCInit(T, layout, &cref);

                computeRefGemm(T, layout, ta, tb, alpha, a, b, @as(T, 1), cref, &cref);
                blazt.ops.gemm(T, layout, ta, tb, alpha, a, b, @as(T, 1), &c);
                try expectMatApproxEq(T, layout, cref, c, tol);
            }
        }
    }
}


