const std = @import("std");
const types = @import("types.zig");
const matrix = @import("matrix.zig");

pub const parallel = struct {
    pub fn gemm(
        comptime T: type,
        trans_a: types.Trans,
        trans_b: types.Trans,
        alpha: T,
        a: matrix.Matrix(T, .row_major), // placeholder; generalized later
        b: matrix.Matrix(T, .row_major), // placeholder; generalized later
        beta: T,
        c: *matrix.Matrix(T, .row_major), // placeholder; generalized later
        thread_pool: anytype,
    ) void {
        _ = .{ T, trans_a, trans_b, alpha, a, b, beta, c, thread_pool };
        @panic("TODO: parallel.gemm");
    }
};


