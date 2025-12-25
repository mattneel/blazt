const std = @import("std");
const blazt = @import("blazt");

const ComplexF32 = std.math.Complex(f32);

fn saxpyLike(n: usize, alpha: f32, x: []const f32, y: []f32) void {
    for (0..n) |i| {
        y[i] = alpha * x[i] + y[i];
    }
    std.mem.doNotOptimizeAway(y[0]);
}

fn axpyOps(n: usize, alpha: f32, x: []const f32, y: []f32) void {
    blazt.ops.axpy(f32, n, alpha, x, y);
    std.mem.doNotOptimizeAway(y[0]);
}

fn dotOps(n: usize, x: []const f32, y: []const f32) void {
    const r = blazt.ops.dot(f32, x[0..n], y[0..n]);
    std.mem.doNotOptimizeAway(&r);
}

fn asumOps(n: usize, x: []const f32) void {
    const r = blazt.ops.asum(f32, x[0..n]);
    std.mem.doNotOptimizeAway(&r);
}

fn nrm2Ops(n: usize, x: []const f32) void {
    const r = blazt.ops.nrm2(f32, x[0..n]);
    std.mem.doNotOptimizeAway(&r);
}

fn iamaxOps(n: usize, x: []const f32) void {
    const idx: usize = blazt.ops.iamax(f32, x[0..n]) orelse 0;
    std.mem.doNotOptimizeAway(&idx);
}

fn iaminOps(n: usize, x: []const f32) void {
    const idx: usize = blazt.ops.iamin(f32, x[0..n]) orelse 0;
    std.mem.doNotOptimizeAway(&idx);
}

fn gemvOps(m: usize, n: usize, a: blazt.Matrix(f32, .row_major), x: []const f32, y: []f32) void {
    blazt.ops.gemv(f32, .row_major, .no_trans, m, n, 1.0, a, x, 0.0, y);
    std.mem.doNotOptimizeAway(y[0]);
}

fn gerOps(m: usize, n: usize, a: *blazt.Matrix(f32, .row_major), x: []const f32, y: []const f32) void {
    blazt.ops.ger(f32, .row_major, m, n, 0.5, x, y, a);
    std.mem.doNotOptimizeAway(a.data[0]);
}

fn trmvOps(n: usize, a: blazt.Matrix(f32, .row_major), x: []f32) void {
    blazt.ops.trmv(f32, .row_major, .upper, .no_trans, .unit, n, a, x);
    std.mem.doNotOptimizeAway(x[0]);
}

fn trsvOps(n: usize, a: blazt.Matrix(f32, .row_major), x: []f32) void {
    // ignore singular in bench setup (we build a well-conditioned matrix)
    _ = blazt.ops.trsv(f32, .row_major, .upper, .no_trans, .unit, n, a, x) catch {};
    std.mem.doNotOptimizeAway(x[0]);
}

fn symvOps(n: usize, a: blazt.Matrix(f32, .row_major), x: []const f32, y: []f32) void {
    blazt.ops.symv(f32, .row_major, .upper, n, 1.0, a, x, 0.0, y);
    std.mem.doNotOptimizeAway(y[0]);
}

fn hemvOps(n: usize, a: blazt.Matrix(ComplexF32, .row_major), x: []const ComplexF32, y: []ComplexF32) void {
    blazt.ops.hemv(ComplexF32, .row_major, .upper, n, ComplexF32.init(1, 0), a, x, ComplexF32.init(0, 0), y);
    std.mem.doNotOptimizeAway(y[0].re);
    std.mem.doNotOptimizeAway(y[0].im);
}

fn copyOps(n: usize, x: []const f32, y: []f32) void {
    blazt.ops.copy(f32, n, x, y);
    std.mem.doNotOptimizeAway(y[0]);
}

fn scalOps(n: usize, alpha: f32, x: []f32) void {
    blazt.ops.scal(f32, n, alpha, x);
    std.mem.doNotOptimizeAway(x[0]);
}

fn swapOps(n: usize, x: []f32, y: []f32) void {
    blazt.ops.swap(f32, n, x, y);
    std.mem.doNotOptimizeAway(x[0]);
    std.mem.doNotOptimizeAway(y[0]);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    const n: usize = 1 << 16;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const y = try alloc.alloc(f32, n);
    defer alloc.free(y);

    // Deterministic init.
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (y, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res = try blazt.bench.run(alloc, "saxpyLike", .{
        .warmup_iters = 5,
        .samples = 30,
        .inner_iters = 3,
    }, saxpyLike, .{ n, @as(f32, 0.5), x, y });
    defer res.deinit();

    res.sortInPlace();

    const p50 = blazt.bench.medianSortedNs(res.samples_ns);
    const p90 = blazt.bench.percentileSortedNs(res.samples_ns, 0.90);
    const p99 = blazt.bench.percentileSortedNs(res.samples_ns, 0.99);

    // A fake-ish "flop count" for this op: 2 flops per element.
    const flops: u64 = @intCast(2 * n);

    var stdout_buffer: [0x400]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const out = &stdout_writer.interface;
    try out.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res.name,
            p50, blazt.bench.gflops(flops, p50),
            p90, blazt.bench.gflops(flops, p90),
            p99, blazt.bench.gflops(flops, p99),
        },
    );
    try out.flush();

    // Large-N axpy benchmark.
    const n_axpy: usize = 1 << 20; // 1M f32 (~4 MiB)
    const x5 = try alloc.alloc(f32, n_axpy);
    defer alloc.free(x5);
    const y5 = try alloc.alloc(f32, n_axpy);
    defer alloc.free(y5);

    for (x5, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (y5, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_axpy = try blazt.bench.run(alloc, "ops.axpy(f32)", .{
        .warmup_iters = 5,
        .samples = 20,
        .inner_iters = 10,
    }, axpyOps, .{ n_axpy, @as(f32, 0.5), x5, y5 });
    defer res_axpy.deinit();

    res_axpy.sortInPlace();
    const a_p50 = blazt.bench.medianSortedNs(res_axpy.samples_ns);
    const a_p90 = blazt.bench.percentileSortedNs(res_axpy.samples_ns, 0.90);
    const a_p99 = blazt.bench.percentileSortedNs(res_axpy.samples_ns, 0.99);

    // 2 flops per element (mul + add).
    const axpy_flops: u64 = @intCast(2 * n_axpy);
    const out_axpy = &stdout_writer.interface;
    try out_axpy.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_axpy.name,
            a_p50, blazt.bench.gflops(axpy_flops, a_p50),
            a_p90, blazt.bench.gflops(axpy_flops, a_p90),
            a_p99, blazt.bench.gflops(axpy_flops, a_p99),
        },
    );
    try out_axpy.flush();

    // Large-N dot benchmark.
    const n_dot: usize = 1 << 20; // 1M f32
    const x6 = try alloc.alloc(f32, n_dot);
    defer alloc.free(x6);
    const y6 = try alloc.alloc(f32, n_dot);
    defer alloc.free(y6);

    for (x6, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (y6, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_dot = try blazt.bench.run(alloc, "ops.dot(f32)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, dotOps, .{ n_dot, x6, y6 });
    defer res_dot.deinit();

    res_dot.sortInPlace();
    const d_p50 = blazt.bench.medianSortedNs(res_dot.samples_ns);
    const d_p90 = blazt.bench.percentileSortedNs(res_dot.samples_ns, 0.90);
    const d_p99 = blazt.bench.percentileSortedNs(res_dot.samples_ns, 0.99);

    const dot_flops: u64 = @intCast(2 * n_dot);
    const out_dot = &stdout_writer.interface;
    try out_dot.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_dot.name,
            d_p50, blazt.bench.gflops(dot_flops, d_p50),
            d_p90, blazt.bench.gflops(dot_flops, d_p90),
            d_p99, blazt.bench.gflops(dot_flops, d_p99),
        },
    );
    try out_dot.flush();

    // GEMV benchmark (row_major, no_trans, beta=0).
    const m_gemv: usize = 1024;
    const n_gemv: usize = 1024;
    var a_gemv = try blazt.Matrix(f32, .row_major).init(alloc, m_gemv, n_gemv);
    defer a_gemv.deinit();
    const x_gemv = try alloc.alloc(f32, n_gemv);
    defer alloc.free(x_gemv);
    const y_gemv = try alloc.alloc(f32, m_gemv);
    defer alloc.free(y_gemv);

    // Deterministic init.
    for (a_gemv.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    }
    for (x_gemv, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    }
    @memset(y_gemv, 0);

    var res_gemv = try blazt.bench.run(alloc, "ops.gemv(f32,row_major)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, gemvOps, .{ m_gemv, n_gemv, a_gemv, x_gemv, y_gemv });
    defer res_gemv.deinit();

    res_gemv.sortInPlace();
    const gv_p50 = blazt.bench.medianSortedNs(res_gemv.samples_ns);
    const gv_p90 = blazt.bench.percentileSortedNs(res_gemv.samples_ns, 0.90);
    const gv_p99 = blazt.bench.percentileSortedNs(res_gemv.samples_ns, 0.99);

    const gemv_flops: u64 = @intCast(2 * m_gemv * n_gemv);
    const out_gemv = &stdout_writer.interface;
    try out_gemv.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_gemv.name,
            gv_p50, blazt.bench.gflops(gemv_flops, gv_p50),
            gv_p90, blazt.bench.gflops(gemv_flops, gv_p90),
            gv_p99, blazt.bench.gflops(gemv_flops, gv_p99),
        },
    );
    try out_gemv.flush();

    // GER benchmark (row_major).
    const m_ger: usize = 512;
    const n_ger: usize = 512;
    var a_ger = try blazt.Matrix(f32, .row_major).init(alloc, m_ger, n_ger);
    defer a_ger.deinit();
    const x_ger = try alloc.alloc(f32, m_ger);
    defer alloc.free(x_ger);
    const y_ger = try alloc.alloc(f32, n_ger);
    defer alloc.free(y_ger);

    @memset(a_ger.data, 0);
    for (x_ger, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (y_ger, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_ger = try blazt.bench.run(alloc, "ops.ger(f32,row_major)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, gerOps, .{ m_ger, n_ger, &a_ger, x_ger, y_ger });
    defer res_ger.deinit();

    res_ger.sortInPlace();
    const gr_p50 = blazt.bench.medianSortedNs(res_ger.samples_ns);
    const gr_p90 = blazt.bench.percentileSortedNs(res_ger.samples_ns, 0.90);
    const gr_p99 = blazt.bench.percentileSortedNs(res_ger.samples_ns, 0.99);

    const ger_flops: u64 = @intCast(2 * m_ger * n_ger);
    const out_ger = &stdout_writer.interface;
    try out_ger.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_ger.name,
            gr_p50, blazt.bench.gflops(ger_flops, gr_p50),
            gr_p90, blazt.bench.gflops(ger_flops, gr_p90),
            gr_p99, blazt.bench.gflops(ger_flops, gr_p99),
        },
    );
    try out_ger.flush();

    // TRMV/TRSV benchmarks (upper, unit diagonal, near-identity with small superdiagonal).
    const n_tri: usize = 1024;
    var a_tri = try blazt.Matrix(f32, .row_major).init(alloc, n_tri, n_tri);
    defer a_tri.deinit();
    const x_tri = try alloc.alloc(f32, n_tri);
    defer alloc.free(x_tri);

    // A = I + 0.001 * superdiag, stable under repeated application.
    @memset(a_tri.data, 0);
    for (0..n_tri) |i| {
        a_tri.data[i * a_tri.stride + i] = 1.0;
        if (i + 1 < n_tri) a_tri.data[i * a_tri.stride + (i + 1)] = 0.001;
    }
    for (x_tri, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);

    var res_trmv = try blazt.bench.run(alloc, "ops.trmv(f32,upper,unit)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, trmvOps, .{ n_tri, a_tri, x_tri });
    defer res_trmv.deinit();

    res_trmv.sortInPlace();
    const tm_p50 = blazt.bench.medianSortedNs(res_trmv.samples_ns);
    const tm_p90 = blazt.bench.percentileSortedNs(res_trmv.samples_ns, 0.90);
    const tm_p99 = blazt.bench.percentileSortedNs(res_trmv.samples_ns, 0.99);
    const trmv_flops: u64 = @intCast(n_tri * (n_tri - 1)); // ~2*(n(n-1)/2) for superdiag-only

    const out_trmv = &stdout_writer.interface;
    try out_trmv.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_trmv.name,
            tm_p50, blazt.bench.gflops(trmv_flops, tm_p50),
            tm_p90, blazt.bench.gflops(trmv_flops, tm_p90),
            tm_p99, blazt.bench.gflops(trmv_flops, tm_p99),
        },
    );
    try out_trmv.flush();

    // Reset x to a bounded range before trsv.
    for (x_tri, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);

    var res_trsv = try blazt.bench.run(alloc, "ops.trsv(f32,upper,unit)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, trsvOps, .{ n_tri, a_tri, x_tri });
    defer res_trsv.deinit();

    res_trsv.sortInPlace();
    const ts_p50 = blazt.bench.medianSortedNs(res_trsv.samples_ns);
    const ts_p90 = blazt.bench.percentileSortedNs(res_trsv.samples_ns, 0.90);
    const ts_p99 = blazt.bench.percentileSortedNs(res_trsv.samples_ns, 0.99);
    const trsv_flops: u64 = trmv_flops; // similar order of work for this near-identity upper system

    const out_trsv = &stdout_writer.interface;
    try out_trsv.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_trsv.name,
            ts_p50, blazt.bench.gflops(trsv_flops, ts_p50),
            ts_p90, blazt.bench.gflops(trsv_flops, ts_p90),
            ts_p99, blazt.bench.gflops(trsv_flops, ts_p99),
        },
    );
    try out_trsv.flush();

    // SYMV benchmark (upper, row_major, beta=0).
    const n_symv: usize = 1024;
    var a_symv = try blazt.Matrix(f32, .row_major).init(alloc, n_symv, n_symv);
    defer a_symv.deinit();
    const x_symv = try alloc.alloc(f32, n_symv);
    defer alloc.free(x_symv);
    const y_symv = try alloc.alloc(f32, n_symv);
    defer alloc.free(y_symv);

    // Deterministic symmetric init.
    @memset(a_symv.data, 0);
    for (0..n_symv) |i| {
        for (i..n_symv) |j| {
            const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 131 + j * 17) % 1024)))) * @as(f32, 0.001);
            a_symv.data[i * a_symv.stride + j] = v;
            a_symv.data[j * a_symv.stride + i] = v;
        }
    }
    for (x_symv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    @memset(y_symv, 0);

    var res_symv = try blazt.bench.run(alloc, "ops.symv(f32,upper,row_major)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, symvOps, .{ n_symv, a_symv, x_symv, y_symv });
    defer res_symv.deinit();

    res_symv.sortInPlace();
    const sy_p50 = blazt.bench.medianSortedNs(res_symv.samples_ns);
    const sy_p90 = blazt.bench.percentileSortedNs(res_symv.samples_ns, 0.90);
    const sy_p99 = blazt.bench.percentileSortedNs(res_symv.samples_ns, 0.99);

    const symv_flops: u64 = @intCast(2 * n_symv * n_symv);
    const out_symv = &stdout_writer.interface;
    try out_symv.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_symv.name,
            sy_p50, blazt.bench.gflops(symv_flops, sy_p50),
            sy_p90, blazt.bench.gflops(symv_flops, sy_p90),
            sy_p99, blazt.bench.gflops(symv_flops, sy_p99),
        },
    );
    try out_symv.flush();

    // HEMV benchmark (upper, row_major, beta=0) on Complex(f32).
    const n_hemv: usize = 512;
    var a_hemv = try blazt.Matrix(ComplexF32, .row_major).init(alloc, n_hemv, n_hemv);
    defer a_hemv.deinit();
    const x_hemv = try alloc.alloc(ComplexF32, n_hemv);
    defer alloc.free(x_hemv);
    const y_hemv = try alloc.alloc(ComplexF32, n_hemv);
    defer alloc.free(y_hemv);

    @memset(a_hemv.data, ComplexF32.init(0, 0));
    for (0..n_hemv) |i| {
        // real diagonal
        a_hemv.data[i * a_hemv.stride + i] = ComplexF32.init(1.0, 0.0);
        for (i + 1..n_hemv) |j| {
            const re = @as(f32, @floatFromInt(@as(u32, @intCast((i * 73 + j * 19) % 1024)))) * @as(f32, 0.001);
            const im = @as(f32, @floatFromInt(@as(u32, @intCast((i * 37 + j * 29) % 1024)))) * @as(f32, 0.001);
            const v = ComplexF32.init(re, im);
            // store upper; also fill lower with conjugate (not required, but keeps matrix well-formed)
            a_hemv.data[i * a_hemv.stride + j] = v;
            a_hemv.data[j * a_hemv.stride + i] = v.conjugate();
        }
    }
    for (x_hemv, 0..) |*v, i| {
        const re = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
        const im = @as(f32, @floatFromInt(@as(u32, @intCast((i * 3) % 1024)))) * @as(f32, 0.001);
        v.* = ComplexF32.init(re, im);
    }
    @memset(y_hemv, ComplexF32.init(0, 0));

    var res_hemv = try blazt.bench.run(alloc, "ops.hemv(ComplexF32,upper,row_major)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, hemvOps, .{ n_hemv, a_hemv, x_hemv, y_hemv });
    defer res_hemv.deinit();

    res_hemv.sortInPlace();
    const hy_p50 = blazt.bench.medianSortedNs(res_hemv.samples_ns);
    const hy_p90 = blazt.bench.percentileSortedNs(res_hemv.samples_ns, 0.90);
    const hy_p99 = blazt.bench.percentileSortedNs(res_hemv.samples_ns, 0.99);

    const hemv_flops: u64 = @intCast(2 * n_hemv * n_hemv);
    const out_hemv = &stdout_writer.interface;
    try out_hemv.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_hemv.name,
            hy_p50, blazt.bench.gflops(hemv_flops, hy_p50),
            hy_p90, blazt.bench.gflops(hemv_flops, hy_p90),
            hy_p99, blazt.bench.gflops(hemv_flops, hy_p99),
        },
    );
    try out_hemv.flush();

    // Large-N asum benchmark.
    const n_asum: usize = 1 << 20; // 1M f32
    const x7 = try alloc.alloc(f32, n_asum);
    defer alloc.free(x7);
    for (x7, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
        v.* = if ((i & 1) == 0) base else -base;
    }

    var res_asum = try blazt.bench.run(alloc, "ops.asum(f32)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, asumOps, .{ n_asum, x7 });
    defer res_asum.deinit();

    res_asum.sortInPlace();
    const as_p50 = blazt.bench.medianSortedNs(res_asum.samples_ns);
    const as_p90 = blazt.bench.percentileSortedNs(res_asum.samples_ns, 0.90);
    const as_p99 = blazt.bench.percentileSortedNs(res_asum.samples_ns, 0.99);

    const asum_bytes: u64 = @intCast(@as(u64, n_asum) * 4); // 1 load per element
    const out_asum = &stdout_writer.interface;
    try out_asum.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_asum.name,
            as_p50, blazt.bench.gibPerSec(asum_bytes, as_p50),
            as_p90, blazt.bench.gibPerSec(asum_bytes, as_p90),
            as_p99, blazt.bench.gibPerSec(asum_bytes, as_p99),
        },
    );
    try out_asum.flush();

    // Large-N nrm2 benchmark.
    var res_nrm2 = try blazt.bench.run(alloc, "ops.nrm2(f32)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, nrm2Ops, .{ n_asum, x7 });
    defer res_nrm2.deinit();

    res_nrm2.sortInPlace();
    const nr_p50 = blazt.bench.medianSortedNs(res_nrm2.samples_ns);
    const nr_p90 = blazt.bench.percentileSortedNs(res_nrm2.samples_ns, 0.90);
    const nr_p99 = blazt.bench.percentileSortedNs(res_nrm2.samples_ns, 0.99);

    const out_nrm2 = &stdout_writer.interface;
    try out_nrm2.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_nrm2.name,
            nr_p50, blazt.bench.gibPerSec(asum_bytes, nr_p50),
            nr_p90, blazt.bench.gibPerSec(asum_bytes, nr_p90),
            nr_p99, blazt.bench.gibPerSec(asum_bytes, nr_p99),
        },
    );
    try out_nrm2.flush();

    // Large-N iamax / iamin benchmarks.
    var res_iamax = try blazt.bench.run(alloc, "ops.iamax(f32)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, iamaxOps, .{ n_asum, x7 });
    defer res_iamax.deinit();

    res_iamax.sortInPlace();
    const imax_p50 = blazt.bench.medianSortedNs(res_iamax.samples_ns);
    const imax_p90 = blazt.bench.percentileSortedNs(res_iamax.samples_ns, 0.90);
    const imax_p99 = blazt.bench.percentileSortedNs(res_iamax.samples_ns, 0.99);

    const out_iamax = &stdout_writer.interface;
    try out_iamax.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_iamax.name,
            imax_p50, blazt.bench.gibPerSec(asum_bytes, imax_p50),
            imax_p90, blazt.bench.gibPerSec(asum_bytes, imax_p90),
            imax_p99, blazt.bench.gibPerSec(asum_bytes, imax_p99),
        },
    );
    try out_iamax.flush();

    var res_iamin = try blazt.bench.run(alloc, "ops.iamin(f32)", .{
        .warmup_iters = 3,
        .samples = 20,
        .inner_iters = 1,
    }, iaminOps, .{ n_asum, x7 });
    defer res_iamin.deinit();

    res_iamin.sortInPlace();
    const imin_p50 = blazt.bench.medianSortedNs(res_iamin.samples_ns);
    const imin_p90 = blazt.bench.percentileSortedNs(res_iamin.samples_ns, 0.90);
    const imin_p99 = blazt.bench.percentileSortedNs(res_iamin.samples_ns, 0.99);

    const out_iamin = &stdout_writer.interface;
    try out_iamin.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_iamin.name,
            imin_p50, blazt.bench.gibPerSec(asum_bytes, imin_p50),
            imin_p90, blazt.bench.gibPerSec(asum_bytes, imin_p90),
            imin_p99, blazt.bench.gibPerSec(asum_bytes, imin_p99),
        },
    );
    try out_iamin.flush();

    // Large-N copy benchmark (memory bandwidth oriented).
    const n_copy: usize = 1 << 20; // 1M f32 (~4 MiB)
    const x2 = try alloc.alloc(f32, n_copy);
    defer alloc.free(x2);
    const y2 = try alloc.alloc(f32, n_copy);
    defer alloc.free(y2);

    for (x2, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    @memset(y2, 0);

    var res_copy = try blazt.bench.run(alloc, "ops.copy(f32)", .{
        .warmup_iters = 5,
        .samples = 20,
        .inner_iters = 20,
    }, copyOps, .{ n_copy, x2, y2 });
    defer res_copy.deinit();

    res_copy.sortInPlace();
    const c_p50 = blazt.bench.medianSortedNs(res_copy.samples_ns);
    const c_p90 = blazt.bench.percentileSortedNs(res_copy.samples_ns, 0.90);
    const c_p99 = blazt.bench.percentileSortedNs(res_copy.samples_ns, 0.99);

    // Copy is ~1 store + 1 load per element => ~8 bytes/elem for f32 (per iteration).
    const bytes: u64 = @intCast(@as(u64, n_copy) * 8);
    const out2 = &stdout_writer.interface;
    try out2.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_copy.name,
            c_p50, blazt.bench.gibPerSec(bytes, c_p50),
            c_p90, blazt.bench.gibPerSec(bytes, c_p90),
            c_p99, blazt.bench.gibPerSec(bytes, c_p99),
        },
    );
    try out2.flush();

    // Large-N scal benchmark (memory bandwidth + mul throughput).
    const n_scal: usize = 1 << 20; // 1M f32 (~4 MiB)
    const x3 = try alloc.alloc(f32, n_scal);
    defer alloc.free(x3);

    for (x3, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);

    // Keep inputs stable across iterations: multiply by -1 flips sign (no drift/denorms).
    var alpha_scal: f32 = -1.0;
    std.mem.doNotOptimizeAway(&alpha_scal);

    var res_scal = try blazt.bench.run(alloc, "ops.scal(f32)", .{
        .warmup_iters = 5,
        .samples = 20,
        .inner_iters = 20,
    }, scalOps, .{ n_scal, alpha_scal, x3 });
    defer res_scal.deinit();

    res_scal.sortInPlace();
    const s_p50 = blazt.bench.medianSortedNs(res_scal.samples_ns);
    const s_p90 = blazt.bench.percentileSortedNs(res_scal.samples_ns, 0.90);
    const s_p99 = blazt.bench.percentileSortedNs(res_scal.samples_ns, 0.99);

    // scal is 1 load + 1 store per element => ~8 bytes/elem for f32 (per iteration).
    const scal_bytes: u64 = @intCast(@as(u64, n_scal) * 8);
    const out3 = &stdout_writer.interface;
    try out3.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_scal.name,
            s_p50, blazt.bench.gibPerSec(scal_bytes, s_p50),
            s_p90, blazt.bench.gibPerSec(scal_bytes, s_p90),
            s_p99, blazt.bench.gibPerSec(scal_bytes, s_p99),
        },
    );
    try out3.flush();

    // Large-N swap benchmark (memory bandwidth oriented).
    const n_swap: usize = 1 << 20; // 1M f32 (~4 MiB) ×2 arrays
    const x4 = try alloc.alloc(f32, n_swap);
    defer alloc.free(x4);
    const y4 = try alloc.alloc(f32, n_swap);
    defer alloc.free(y4);

    for (x4, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (y4, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_swap = try blazt.bench.run(alloc, "ops.swap(f32)", .{
        .warmup_iters = 5,
        .samples = 20,
        .inner_iters = 20,
    }, swapOps, .{ n_swap, x4, y4 });
    defer res_swap.deinit();

    res_swap.sortInPlace();
    const w_p50 = blazt.bench.medianSortedNs(res_swap.samples_ns);
    const w_p90 = blazt.bench.percentileSortedNs(res_swap.samples_ns, 0.90);
    const w_p99 = blazt.bench.percentileSortedNs(res_swap.samples_ns, 0.99);

    // swap is 2 loads + 2 stores per element => ~16 bytes/elem for f32 (per iteration).
    const swap_bytes: u64 = @intCast(@as(u64, n_swap) * 16);
    const out4 = &stdout_writer.interface;
    try out4.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GiB/s)\n  p90: {d} ns  ({d:.3} GiB/s)\n  p99: {d} ns  ({d:.3} GiB/s)\n",
        .{
            res_swap.name,
            w_p50, blazt.bench.gibPerSec(swap_bytes, w_p50),
            w_p90, blazt.bench.gibPerSec(swap_bytes, w_p90),
            w_p99, blazt.bench.gibPerSec(swap_bytes, w_p99),
        },
    );
    try out4.flush();
}


