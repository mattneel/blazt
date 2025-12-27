//! Parallel BLAS routines built on top of `blazt.ThreadPool`.
//!
//! The API mirrors `blazt.ops` where applicable, but accepts a thread pool to schedule work.

const std = @import("std");
const types = @import("types.zig");
const matrix = @import("matrix.zig");
const memory = @import("memory.zig");
const gemm_mod = @import("gemm.zig");
const ops = @import("ops.zig").ops;
const thread_pool_mod = @import("thread_pool.zig");

pub const parallel = struct {
    /// Parallel GEMM: `C := alpha*A*B + beta*C`.
    ///
    /// Currently specialized for:
    /// - `layout = .row_major`
    /// - `trans_a = .no_trans`, `trans_b = .no_trans` (other modes fall back to `ops.gemm`)
    ///
    /// Work is split across column panels to avoid write contention.
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

        gemmParallelBlockedRowMajor(T, m, n, k, alpha, a, b, beta, c, thread_pool);
    }
};

fn gemmParallelBlockedRowMajor(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: matrix.Matrix(T, .row_major),
    b: matrix.Matrix(T, .row_major),
    beta: T,
    c: *matrix.Matrix(T, .row_major),
    thread_pool: *thread_pool_mod.ThreadPool,
) void {
    const P: gemm_mod.TileParams = comptime gemm_mod.computeTileParamsRowMajor(T);
    const MR: usize = P.MR;
    const NR: usize = P.NR;
    const KC: usize = P.KC;
    const MC: usize = P.MC;
    const NC: usize = P.NC;

    if (m == 0 or n == 0) return;

    const pool_threads: usize = thread_pool.threads.len;
    if (pool_threads <= 1) {
        ops.gemm(T, .row_major, .no_trans, .no_trans, alpha, a, b, beta, c);
        return;
    }

    const PB_STRIDE_ELEMS: usize = comptime gemm_mod.packBPanelStrideElems(T, P);
    const PACK_B_STACK_PANELS: usize = (NC + NR - 1) / NR;
    const PACK_B_STACK_ELEMS: usize = PB_STRIDE_ELEMS * @max(PACK_B_STACK_PANELS, 1);

    // Shared packed-B buffer (per call). Packed panels are cache-line aligned via PB_STRIDE_ELEMS.
    var pack_b_shared: [PACK_B_STACK_ELEMS]T align(memory.CacheLine) = undefined;

    // Adaptive MC for parallelism: when M is small, reduce MC so we have enough row blocks
    // for threads to work on (avoids under-parallelization when MC is large).
    const mc_step: usize = blk: {
        const want = (m + pool_threads - 1) / pool_threads;
        const want_aligned = roundUpMultiple(@max(want, MR), MR);
        break :blk @min(MC, want_aligned);
    };

    const WorkerCtx = struct {
        const Self = @This();

        // Shared panel parameters (written by main thread before dispatch)
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
        a: matrix.Matrix(T, .row_major),
        b: matrix.Matrix(T, .row_major),
        c: *matrix.Matrix(T, .row_major),
        mc_step: usize,
        jc: usize,
        nc_cur: usize,
        pc: usize,
        kc_cur: usize,
        beta_block: T,
        pack_b: [*]T,
        pb_panels: usize,
        next_panel: *std.atomic.Value(usize),
        pack_done: *std.atomic.Value(usize),
        task_count: usize,
        next_block: *std.atomic.Value(usize),
        total_blocks: usize,

        fn runOpaque(ctx_opaque: *anyopaque) void {
            const ctx: *Self = @ptrCast(@alignCast(ctx_opaque));
            ctx.run();
        }

        fn run(self: *Self) void {
            // Per-worker packed A buffer (stack-local).
            const PACK_A_BLOCK_ELEMS: usize = MC * KC;
            var pack_a_buf: [PACK_A_BLOCK_ELEMS]T align(memory.CacheLine) = undefined;

            // Phase 1: cooperatively pack B panels for this (jc,pc) into the shared buffer.
            while (true) {
                const jp = self.next_panel.fetchAdd(1, .acq_rel);
                if (jp >= self.pb_panels) break;

                const jr: usize = jp * NR;
                const nr_cur = @min(NR, self.nc_cur - jr);

                const pb_ptr_unaligned: [*]T = self.pack_b + jp * PB_STRIDE_ELEMS;
                const pb_ptr: [*]align(memory.CacheLine) T = @ptrCast(@alignCast(pb_ptr_unaligned));
                const pb: []align(memory.CacheLine) T = pb_ptr[0 .. NR * self.kc_cur];
                gemm_mod.packB(T, .row_major, NR, self.kc_cur, nr_cur, self.b, self.pc, self.jc + jr, pb);
            }

            // Barrier: ensure packed-B is fully written before any thread starts computing.
            _ = self.pack_done.fetchAdd(1, .acq_rel);
            while (self.pack_done.load(.acquire) < self.task_count) {
                std.atomic.spinLoopHint();
            }

            while (true) {
                const bi = self.next_block.fetchAdd(1, .acq_rel);
                if (bi >= self.total_blocks) break;

                const ic: usize = bi * self.mc_step;
                if (ic >= self.m) continue;
                const mc_cur = @min(self.mc_step, self.m - ic);
                const mc_pad = roundUpMultiple(mc_cur, MR);
                const pack_a_block: []align(memory.CacheLine) T = pack_a_buf[0 .. mc_pad * self.kc_cur];

                // Pack A block (row_major): layout matches `gemmBlockedRangeRowMajor` block packing.
                var ir_pack: usize = 0;
                while (ir_pack < mc_pad) : (ir_pack += MR) {
                    const row_base = ic + ir_pack;
                    for (0..self.kc_cur) |p| {
                        const src_col: usize = self.pc + p;
                        const dst_base = ir_pack * self.kc_cur + p * MR;
                        inline for (0..MR) |idx_i| {
                            const r = row_base + idx_i;
                            pack_a_block[dst_base + idx_i] = if (r < ic + mc_cur) self.a.data[r * self.a.stride + src_col] else @as(T, 0);
                        }
                    }
                }

                // Compute this (ic,jc) block using shared packed B.
                var jp: usize = 0;
                while (jp < self.pb_panels) : (jp += 1) {
                    const jr: usize = jp * NR;
                    const nr_cur = @min(NR, self.nc_cur - jr);
                    const pb_ptr: [*]const T = self.pack_b + jp * PB_STRIDE_ELEMS;

                    var ir: usize = 0;
                    while (ir < mc_cur) : (ir += MR) {
                        const mr_cur = @min(MR, mc_cur - ir);
                        const pa_ptr: [*]const T = pack_a_block.ptr + ir * self.kc_cur;
                        const c_ptr: [*]T = self.c.data[(ic + ir) * self.c.stride + (self.jc + jr) ..].ptr;

                        if (mr_cur == MR and nr_cur == NR) {
                            gemm_mod.microKernelRowMajor(T, MR, NR, self.kc_cur, pa_ptr, pb_ptr, c_ptr, self.c.stride, 1, self.alpha, self.beta_block);
                        } else {
                            gemm_mod.microKernelRowMajorPartial(T, MR, NR, self.kc_cur, pa_ptr, pb_ptr, c_ptr, self.c.stride, 1, self.alpha, self.beta_block, mr_cur, nr_cur);
                        }
                    }
                }
            }
        }
    };

    var ctxs: [256]WorkerCtx = undefined;

    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc_cur = @min(NC, n - jc);

        var pc: usize = 0;
        while (pc < k) : (pc += KC) {
            const kc_cur = @min(KC, k - pc);
            const beta_block: T = if (pc == 0) beta else @as(T, 1);

            const pb_panels: usize = (nc_cur + NR - 1) / NR;
            std.debug.assert(pb_panels <= PACK_B_STACK_PANELS);

            // Dispatch workers over IC blocks for this panel.
            var next_panel = std.atomic.Value(usize).init(0);
            var pack_done = std.atomic.Value(usize).init(0);
            var next_block = std.atomic.Value(usize).init(0);
            const total_blocks: usize = (m + mc_step - 1) / mc_step;
            const task_count: usize = @min(pool_threads, 256);

            for (0..task_count) |ti| {
                ctxs[ti] = .{
                    .m = m,
                    .n = n,
                    .k = k,
                    .alpha = alpha,
                    .beta = beta,
                    .a = a,
                    .b = b,
                    .c = c,
                    .mc_step = mc_step,
                    .jc = jc,
                    .nc_cur = nc_cur,
                    .pc = pc,
                    .kc_cur = kc_cur,
                    .beta_block = beta_block,
                    .pack_b = pack_b_shared[0..].ptr,
                    .pb_panels = pb_panels,
                    .next_panel = &next_panel,
                    .pack_done = &pack_done,
                    .task_count = task_count,
                    .next_block = &next_block,
                    .total_blocks = total_blocks,
                };

                thread_pool.submit(WorkerCtx.runOpaque, &ctxs[ti]) catch {
                    ctxs[ti].run();
                };
            }

            thread_pool.waitAll();
        }
    }
}

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
        const PACK_A_BLOCK_ELEMS: usize = P.MC * P.KC;
        var pack_a_small: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        const pack_a_big = memory.allocAligned(thread_pool.allocator, T, PACK_A_BLOCK_ELEMS) catch null;
        defer if (pack_a_big) |buf| thread_pool.allocator.free(buf);
        const pack_a_use: []align(memory.CacheLine) T = pack_a_big orelse pack_a_small[0..];

        const PB_STRIDE_ELEMS: usize = comptime gemm_mod.packBPanelStrideElems(T, P);
        const PB_PANELS: usize = comptime gemm_mod.packBPanelCount(T, P);
        var pack_b: [PB_STRIDE_ELEMS * PB_PANELS]T align(memory.CacheLine) = undefined;
        gemm_mod.gemmBlocked(T, m, n, k, alpha, a, b, beta, c, pack_a_use, pack_b[0..]);
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
        const PACK_A_BLOCK_ELEMS: usize = P.MC * P.KC;
        var pack_a_small: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
        const pack_a_big = memory.allocAligned(thread_pool.allocator, T, PACK_A_BLOCK_ELEMS) catch null;
        defer if (pack_a_big) |buf| thread_pool.allocator.free(buf);
        const pack_a_use: []align(memory.CacheLine) T = pack_a_big orelse pack_a_small[0..];

        const PB_STRIDE_ELEMS: usize = comptime gemm_mod.packBPanelStrideElems(T, P);
        const PB_PANELS: usize = comptime gemm_mod.packBPanelCount(T, P);
        var pack_b: [PB_STRIDE_ELEMS * PB_PANELS]T align(memory.CacheLine) = undefined;
        gemm_mod.gemmBlocked(T, m, n, k, alpha, a, b, beta, c, pack_a_use, pack_b[0..]);
        return;
    }

    const TaskCtx = struct {
        const Self = @This();
        allocator: std.mem.Allocator,
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

            const PB_STRIDE_ELEMS: usize = comptime gemm_mod.packBPanelStrideElems(T, P);
            const PB_PANELS: usize = comptime gemm_mod.packBPanelCount(T, P);
            const PACK_A_BLOCK_ELEMS: usize = P.MC * P.KC;
            const PACK_B_STACK_PANELS: usize = (P.NC + P.NR - 1) / P.NR;
            const PACK_B_STACK_ELEMS: usize = PB_STRIDE_ELEMS * @max(PACK_B_STACK_PANELS, 1);

            // Prefer stack pack buffers for performance (avoid allocator contention in hot paths),
            // but cap total stack usage per task to a conservative budget.
            const STACK_MAX_BYTES: usize = 2 * 1024 * 1024;
            if (comptime (PACK_A_BLOCK_ELEMS * @sizeOf(T) + PACK_B_STACK_ELEMS * @sizeOf(T) <= STACK_MAX_BYTES)) {
                var pack_a_stack: [PACK_A_BLOCK_ELEMS]T align(memory.CacheLine) = undefined;
                var pack_b_stack: [PACK_B_STACK_ELEMS]T align(memory.CacheLine) = undefined;

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
                    pack_a_stack[0..],
                    pack_b_stack[0..],
                    self.col0,
                    self.col1,
                );
                return;
            }

            // Fallback: heap pack buffers.
            var pack_a_small: [P.MR * P.KC]T align(memory.CacheLine) = undefined;
            const pack_a_big = memory.allocAligned(self.allocator, T, PACK_A_BLOCK_ELEMS) catch null;
            defer if (pack_a_big) |buf| self.allocator.free(buf);
            const pack_a_use: []align(memory.CacheLine) T = pack_a_big orelse pack_a_small[0..];

            const cols_range: usize = self.col1 - self.col0;
            const nc_cap: usize = @min(P.NC, cols_range);
            const pb_full_panels: usize = (nc_cap + P.NR - 1) / P.NR;
            const pb_panels_alloc: usize = @max(PB_PANELS, @max(pb_full_panels, 1));
            const PACK_B_ALLOC_ELEMS: usize = PB_STRIDE_ELEMS * pb_panels_alloc;

            var pack_b_small: [PB_STRIDE_ELEMS * PB_PANELS]T align(memory.CacheLine) = undefined;
            const PACK_B_MAX_BYTES: usize = 8 * 1024 * 1024;
            const pack_b_big = if (PACK_B_ALLOC_ELEMS * @sizeOf(T) <= PACK_B_MAX_BYTES)
                memory.allocAligned(self.allocator, T, PACK_B_ALLOC_ELEMS) catch null
            else
                null;
            defer if (pack_b_big) |buf| self.allocator.free(buf);
            const pack_b_use: []align(memory.CacheLine) T = pack_b_big orelse pack_b_small[0..];

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
                pack_a_use,
                pack_b_use,
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
            .allocator = thread_pool.allocator,
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

fn roundUpMultiple(x: usize, multiple: usize) usize {
    if (multiple == 0) return x;
    const r = x % multiple;
    return if (r == 0) x else x + (multiple - r);
}


