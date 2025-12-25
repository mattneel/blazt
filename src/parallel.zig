const std = @import("std");
const types = @import("types.zig");
const matrix = @import("matrix.zig");
const memory = @import("memory.zig");
const gemm_mod = @import("gemm.zig");
const ops = @import("ops.zig").ops;
const thread_pool_mod = @import("thread_pool.zig");

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
        thread_pool: *thread_pool_mod.ThreadPool,
    ) void {
        if (comptime @typeInfo(T) != .float) {
            @compileError("parallel.gemm currently only supports floating-point types");
        }

        // For real types, `.conj_trans` is equivalent to `.trans`.
        const eff_a: types.Trans = switch (trans_a) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };
        const eff_b: types.Trans = switch (trans_b) {
            .no_trans => .no_trans,
            .trans, .conj_trans => .trans,
        };

        // Fast path: only specialize for the common no-transpose case for now.
        if (eff_a != .no_trans or eff_b != .no_trans) {
            ops.gemm(T, .row_major, trans_a, trans_b, alpha, a, b, beta, c);
            return;
        }

        // Validate storage invariants (debug only).
        std.debug.assert(a.stride >= a.cols);
        std.debug.assert(b.stride >= b.cols);
        std.debug.assert(c.stride >= c.cols);
        if (a.rows != 0 and a.cols != 0) std.debug.assert((a.rows - 1) * a.stride + a.cols <= a.data.len);
        if (b.rows != 0 and b.cols != 0) std.debug.assert((b.rows - 1) * b.stride + b.cols <= b.data.len);
        if (c.rows != 0 and c.cols != 0) std.debug.assert((c.rows - 1) * c.stride + c.cols <= c.data.len);

        // Dimensions: A is m×k, B is k×n, C is m×n.
        const m: usize = c.rows;
        const n: usize = c.cols;
        std.debug.assert(a.rows == m);
        const k: usize = a.cols;
        std.debug.assert(b.rows == k);
        std.debug.assert(b.cols == n);

        if (m == 0 or n == 0) return;

        // Trivial cases: just reuse the sequential implementation.
        if (k == 0 or alpha == @as(T, 0) or thread_pool.threads.len <= 1) {
            ops.gemm(T, .row_major, .no_trans, .no_trans, alpha, a, b, beta, c);
            return;
        }

        // Row-major GEMM can be expressed as a col-major GEMM on transposed views:
        // (A*B)^T = B^T * A^T
        const a_t: matrix.Matrix(T, .col_major) = .{
            .data = a.data,
            .rows = a.cols,
            .cols = a.rows,
            .stride = a.stride,
            .allocator = a.allocator,
        };
        const b_t: matrix.Matrix(T, .col_major) = .{
            .data = b.data,
            .rows = b.cols,
            .cols = b.rows,
            .stride = b.stride,
            .allocator = b.allocator,
        };
        var c_t: matrix.Matrix(T, .col_major) = .{
            .data = c.data,
            .rows = c.cols,
            .cols = c.rows,
            .stride = c.stride,
            .allocator = c.allocator,
        };

        // Compute Cᵀ := alpha * Bᵀ * Aᵀ + beta * Cᵀ in parallel by splitting column panels of Cᵀ.
        gemmParallelBlockedColMajor(T, n, m, k, alpha, b_t, a_t, beta, &c_t, thread_pool);
    }
};

fn gemmParallelBlockedColMajor(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: matrix.Matrix(T, .col_major),
    b: matrix.Matrix(T, .col_major),
    beta: T,
    c: *matrix.Matrix(T, .col_major),
    thread_pool: *thread_pool_mod.ThreadPool,
) void {
    const P: gemm_mod.TileParams = comptime gemm_mod.computeTileParams(T);
    const NR: usize = P.NR;

    // If we can't meaningfully split, just run sequential.
    if (n == 0 or thread_pool.threads.len <= 1) {
        var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;
        gemm_mod.gemmBlocked(T, m, n, k, alpha, a, b, beta, c, pack_a[0..], pack_b[0..]);
        return;
    }

    const pool_threads: usize = thread_pool.threads.len;
    const max_tasks: usize = @min(pool_threads, 256);

    // Minimum columns per task to avoid overhead. (Aligned to NR for nicer packing.)
    const min_cols_per_task: usize = @max(NR * 16, 64);

    // Choose a chunk width and derived task count.
    const chunk_cols_unaligned: usize = (n + max_tasks - 1) / max_tasks;
    var chunk_cols: usize = chunk_cols_unaligned;
    if (chunk_cols < min_cols_per_task) chunk_cols = min_cols_per_task;
    // Round up to a multiple of NR.
    if (NR != 0) {
        const r = chunk_cols % NR;
        if (r != 0) chunk_cols += (NR - r);
    }

    const task_count: usize = @min(max_tasks, (n + chunk_cols - 1) / chunk_cols);
    if (task_count <= 1) {
        var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;
        gemm_mod.gemmBlocked(T, m, n, k, alpha, a, b, beta, c, pack_a[0..], pack_b[0..]);
        return;
    }

    const TaskCtx = struct {
        const Self = @This();
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
        a: matrix.Matrix(T, .col_major),
        b: matrix.Matrix(T, .col_major),
        c: *matrix.Matrix(T, .col_major),
        col0: usize,
        col1: usize,

        fn runOpaque(ctx_opaque: *anyopaque) void {
            const ctx: *Self = @ptrCast(@alignCast(ctx_opaque));
            ctx.run();
        }

        fn run(self: *Self) void {
            if (self.col1 <= self.col0) return;

            // Per-task pack buffers (stack-local; avoids sharing).
            var pack_a: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
            var pack_b: [P.NR * P.KC]T align(memory.CacheLine) = undefined;

            gemm_mod.gemmBlockedRange(
                T,
                self.m,
                self.n,
                self.k,
                self.alpha,
                self.a,
                self.b,
                self.beta,
                self.c,
                pack_a[0..],
                pack_b[0..],
                self.col0,
                self.col1,
            );
        }
    };

    var ctxs: [256]TaskCtx = undefined;

    for (0..task_count) |ti| {
        const col0: usize = ti * chunk_cols;
        const col1: usize = @min(n, col0 + chunk_cols);

        ctxs[ti] = .{
            .m = m,
            .n = n,
            .k = k,
            .alpha = alpha,
            .beta = beta,
            .a = a,
            .b = b,
            .c = c,
            .col0 = col0,
            .col1 = col1,
        };

        thread_pool.submit(TaskCtx.runOpaque, &ctxs[ti]) catch {
            // Pool saturated: run inline on the caller thread for this chunk.
            ctxs[ti].run();
        };
    }

    // Wait for all submitted tasks to complete.
    thread_pool.waitAll();
}


