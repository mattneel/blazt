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

fn gemmOps(
    m: usize,
    n: usize,
    k: usize,
    a: blazt.Matrix(f32, .row_major),
    b: blazt.Matrix(f32, .row_major),
    c: *blazt.Matrix(f32, .row_major),
) void {
    _ = .{ m, n, k };
    blazt.ops.gemm(f32, .row_major, .no_trans, .no_trans, 1.0, a, b, 0.0, c);
    std.mem.doNotOptimizeAway(c.data[0]);
}

fn syrkOps(
    n: usize,
    k: usize,
    a: blazt.Matrix(f32, .row_major),
    c: *blazt.Matrix(f32, .row_major),
) void {
    blazt.ops.syrk(f32, .row_major, .upper, .no_trans, n, k, 1.0, a, 0.0, c);
    std.mem.doNotOptimizeAway(c.data[0]);
}

fn syr2kOps(
    n: usize,
    k: usize,
    a: blazt.Matrix(f32, .row_major),
    b: blazt.Matrix(f32, .row_major),
    c: *blazt.Matrix(f32, .row_major),
) void {
    blazt.ops.syr2k(f32, .row_major, .upper, .no_trans, n, k, 1.0, a, b, 0.0, c);
    std.mem.doNotOptimizeAway(c.data[0]);
}

fn symmOps(
    m: usize,
    n: usize,
    a: blazt.Matrix(f32, .row_major),
    b: blazt.Matrix(f32, .row_major),
    c: *blazt.Matrix(f32, .row_major),
) void {
    blazt.ops.symm(f32, .row_major, .left, .upper, m, n, 1.0, a, b, 0.0, c);
    std.mem.doNotOptimizeAway(c.data[0]);
}

fn trmmOps(
    m: usize,
    n: usize,
    a: blazt.Matrix(f32, .row_major),
    b0: blazt.Matrix(f32, .row_major),
    b: *blazt.Matrix(f32, .row_major),
) void {
    _ = .{ m, n };
    @memcpy(b.data, b0.data);
    blazt.ops.trmm(f32, .row_major, .left, .upper, .no_trans, .non_unit, b.rows, b.cols, 1.0, a, b);
    std.mem.doNotOptimizeAway(b.data[0]);
}

fn trsmOps(
    m: usize,
    n: usize,
    a: blazt.Matrix(f32, .row_major),
    b0: blazt.Matrix(f32, .row_major),
    b: *blazt.Matrix(f32, .row_major),
) void {
    _ = .{ m, n };
    @memcpy(b.data, b0.data);
    blazt.ops.trsm(f32, .row_major, .left, .upper, .no_trans, .non_unit, b.rows, b.cols, 1.0, a, b) catch unreachable;
    std.mem.doNotOptimizeAway(b.data[0]);
}

fn luOps(
    n: usize,
    a0: blazt.Matrix(f32, .row_major),
    a: *blazt.Matrix(f32, .row_major),
    ipiv: []i32,
) void {
    _ = n;
    @memcpy(a.data, a0.data);
    _ = blazt.ops.lu(f32, a, ipiv) catch unreachable;
    std.mem.doNotOptimizeAway(a.data[0]);
}

fn choleskyOps(
    a0: blazt.Matrix(f32, .row_major),
    a: *blazt.Matrix(f32, .row_major),
) void {
    @memcpy(a.data, a0.data);
    _ = blazt.ops.cholesky(f32, .lower, a) catch unreachable;
    std.mem.doNotOptimizeAway(a.data[0]);
}

fn qrOps(
    m: usize,
    n: usize,
    a0: blazt.Matrix(f32, .row_major),
    a: *blazt.Matrix(f32, .row_major),
    tau: []f32,
) void {
    _ = .{ m, n };
    @memcpy(a.data, a0.data);
    blazt.ops.qr(f32, a, tau);
    std.mem.doNotOptimizeAway(a.data[0]);
}

fn svdOps(
    m: usize,
    n: usize,
    a0: blazt.Matrix(f32, .row_major),
    a: *blazt.Matrix(f32, .row_major),
    s: []f32,
    u: *blazt.Matrix(f32, .row_major),
    vt: *blazt.Matrix(f32, .row_major),
) void {
    _ = .{ m, n };
    @memcpy(a.data, a0.data);
    _ = blazt.ops.svd(f32, a, s, u, vt) catch unreachable;
    std.mem.doNotOptimizeAway(a.data[0]);
}

fn eigOps(
    n: usize,
    a0: blazt.Matrix(f32, .row_major),
    a: *blazt.Matrix(f32, .row_major),
    w: []f32,
    v: *blazt.Matrix(f32, .row_major),
) void {
    _ = n;
    @memcpy(a.data, a0.data);
    _ = blazt.ops.eig(f32, a, w, v) catch unreachable;
    std.mem.doNotOptimizeAway(w[0]);
}

fn gemmParallelOps(
    m: usize,
    n: usize,
    k: usize,
    a: blazt.Matrix(f32, .row_major),
    b: blazt.Matrix(f32, .row_major),
    c: *blazt.Matrix(f32, .row_major),
    pool: *blazt.ThreadPool,
) void {
    _ = .{ m, n, k };
    blazt.parallel.gemm(f32, .no_trans, .no_trans, 1.0, a, b, 0.0, c, pool);
    std.mem.doNotOptimizeAway(c.data[0]);
}

fn gemmOpsColMajor(
    m: usize,
    n: usize,
    k: usize,
    a: blazt.Matrix(f32, .col_major),
    b: blazt.Matrix(f32, .col_major),
    c: *blazt.Matrix(f32, .col_major),
) void {
    _ = .{ m, n, k };
    blazt.ops.gemm(f32, .col_major, .no_trans, .no_trans, 1.0, a, b, 0.0, c);
    std.mem.doNotOptimizeAway(c.data[0]);
}

fn oracleSgemmOps(
    oracle: *const blazt.oracle.Oracle,
    m: usize,
    n: usize,
    k: usize,
    a: blazt.Matrix(f32, .col_major),
    b: blazt.Matrix(f32, .col_major),
    c: *blazt.Matrix(f32, .col_major),
) void {
    _ = .{ m, n, k };
    oracle.sgemm(
        .col_major,
        .no_trans,
        .no_trans,
        @intCast(c.rows),
        @intCast(c.cols),
        @intCast(a.cols),
        1.0,
        a.data.ptr,
        @intCast(a.stride),
        b.data.ptr,
        @intCast(b.stride),
        0.0,
        c.data.ptr,
        @intCast(c.stride),
    );
    std.mem.doNotOptimizeAway(c.data[0]);
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

fn hotSumU64(x: []const u64) u64 {
    var sum: u64 = 0;
    for (x) |v| sum +%= v;
    std.mem.doNotOptimizeAway(sum);
    return sum;
}

fn benchCacheHotAfterZero(allocator: std.mem.Allocator, out: anytype) !void {
    // Cache pollution proxy: time a "hot" working set after a large write-only zeroing pass.
    // Compare runs with/without `-Dnt_stores=true`.
    //
    // The zeroing is executed before the timer starts, so the sample measures only the hot loop.
    const big_bytes: usize = 64 * 1024 * 1024; // 64 MiB
    const hot_bytes: usize = 8 * 1024 * 1024; // 8 MiB
    const hot_len: usize = hot_bytes / @sizeOf(u64);

    const y = try blazt.allocAligned(allocator, u8, big_bytes);
    defer allocator.free(y);
    const hot = try allocator.alloc(u64, hot_len);
    defer allocator.free(hot);

    @memset(y, 0xAA);
    for (hot, 0..) |*v, i| v.* = @as(u64, @intCast(i)) *% 0x9e3779b97f4a7c15;

    // Warm hot set into cache.
    _ = hotSumU64(hot);

    var timer = try std.time.Timer.start();

    const samples: usize = 20;
    var base_ns: [samples]u64 = undefined;
    var after_ns: [samples]u64 = undefined;

    // Baseline hot set timing.
    for (&base_ns) |*t| {
        timer.reset();
        _ = hotSumU64(hot);
        t.* = timer.read();
    }

    // Hot set timing immediately after a large zeroing write.
    for (&after_ns) |*t| {
        blazt.memory.memsetZeroBytes(y);
        timer.reset();
        _ = hotSumU64(hot);
        t.* = timer.read();
    }

    blazt.bench.sortNs(base_ns[0..]);
    blazt.bench.sortNs(after_ns[0..]);
    const base_p50 = blazt.bench.medianSortedNs(base_ns[0..]);
    const after_p50 = blazt.bench.medianSortedNs(after_ns[0..]);

    const ratio: f64 = if (base_p50 == 0)
        std.math.inf(f64)
    else
        @as(f64, @floatFromInt(after_p50)) / @as(f64, @floatFromInt(base_p50));

    try out.print(
        "bench cache_hot_after_zero (nt_stores={})\n  big_zero: {d} bytes\n  hot: {d} bytes\n  hot_p50: {d} ns\n  hot_after_zero_p50: {d} ns\n  ratio: {d:.3}\n",
        .{ blazt.build_options.nt_stores, big_bytes, hot_bytes, base_p50, after_p50, ratio },
    );
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    // Some decompositions (notably Jacobi SVD / Jacobi eig) are intentionally coarse and can take
    // a very long time at larger sizes. Keep the default `zig build bench` run snappy, and
    // allow opting into heavier sizes via env.
    const lapack_heavy: bool = blk: {
        const s = std.process.getEnvVarOwned(alloc, "BLAZT_BENCH_LAPACK_HEAVY") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => break :blk false,
            else => return err,
        };
        defer alloc.free(s);
        break :blk s.len != 0 and !std.mem.eql(u8, s, "0");
    };

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

    // ---------------------------------------------------------------------
    // Sanity: show which cpu_cache data + tile params this build is using.
    // (This helps catch "cpu_cache_default got wired in" regressions.)
    // ---------------------------------------------------------------------
    {
        const info = blazt.CpuInfo.native();
        const p32 = blazt.gemm.computeTileParams(f32);
        const p64 = blazt.gemm.computeTileParams(f64);
        const p32_rm = blazt.gemm.computeTileParamsRowMajor(f32);
        const p64_rm = blazt.gemm.computeTileParamsRowMajor(f64);
        try out.print(
            "info: cpu_cache detected={} method={s} l1d={}B l2={}B l3={}B l1d_share={} l2_share={} l3_share={}\n" ++
                "info: gemm.tile_col_major(f32) MR={} NR={} KC={} MC={} NC={}\n" ++
                "info: gemm.tile_col_major(f64) MR={} NR={} KC={} MC={} NC={}\n" ++
                "info: gemm.tile_row_major(f32) MR={} NR={} KC={} MC={} NC={}\n" ++
                "info: gemm.tile_row_major(f64) MR={} NR={} KC={} MC={} NC={}\n",
            .{
                blazt.cpu.cache.detected,
                blazt.cpu.cache.method,
                info.l1d_size_bytes,
                info.l2_size_bytes,
                info.l3_size_bytes,
                info.l1d_shared_by_logical_cpus,
                info.l2_shared_by_logical_cpus,
                info.l3_shared_by_logical_cpus,
                p32.MR,
                p32.NR,
                p32.KC,
                p32.MC,
                p32.NC,
                p64.MR,
                p64.NR,
                p64.KC,
                p64.MC,
                p64.NC,
                p32_rm.MR,
                p32_rm.NR,
                p32_rm.KC,
                p32_rm.MC,
                p32_rm.NC,
                p64_rm.MR,
                p64_rm.NR,
                p64_rm.KC,
                p64_rm.MC,
                p64_rm.NC,
            },
        );
    }
    try out.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res.name,
            p50,
            blazt.bench.gflops(flops, p50),
            p90,
            blazt.bench.gflops(flops, p90),
            p99,
            blazt.bench.gflops(flops, p99),
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
            a_p50,
            blazt.bench.gflops(axpy_flops, a_p50),
            a_p90,
            blazt.bench.gflops(axpy_flops, a_p90),
            a_p99,
            blazt.bench.gflops(axpy_flops, a_p99),
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
            d_p50,
            blazt.bench.gflops(dot_flops, d_p50),
            d_p90,
            blazt.bench.gflops(dot_flops, d_p90),
            d_p99,
            blazt.bench.gflops(dot_flops, d_p99),
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
            gv_p50,
            blazt.bench.gflops(gemv_flops, gv_p50),
            gv_p90,
            blazt.bench.gflops(gemv_flops, gv_p90),
            gv_p99,
            blazt.bench.gflops(gemv_flops, gv_p99),
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
            gr_p50,
            blazt.bench.gflops(ger_flops, gr_p50),
            gr_p90,
            blazt.bench.gflops(ger_flops, gr_p90),
            gr_p99,
            blazt.bench.gflops(ger_flops, gr_p99),
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
            tm_p50,
            blazt.bench.gflops(trmv_flops, tm_p50),
            tm_p90,
            blazt.bench.gflops(trmv_flops, tm_p90),
            tm_p99,
            blazt.bench.gflops(trmv_flops, tm_p99),
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
            ts_p50,
            blazt.bench.gflops(trsv_flops, ts_p50),
            ts_p90,
            blazt.bench.gflops(trsv_flops, ts_p90),
            ts_p99,
            blazt.bench.gflops(trsv_flops, ts_p99),
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
            sy_p50,
            blazt.bench.gflops(symv_flops, sy_p50),
            sy_p90,
            blazt.bench.gflops(symv_flops, sy_p90),
            sy_p99,
            blazt.bench.gflops(symv_flops, sy_p99),
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
            hy_p50,
            blazt.bench.gflops(hemv_flops, hy_p50),
            hy_p90,
            blazt.bench.gflops(hemv_flops, hy_p90),
            hy_p99,
            blazt.bench.gflops(hemv_flops, hy_p99),
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
            as_p50,
            blazt.bench.gibPerSec(asum_bytes, as_p50),
            as_p90,
            blazt.bench.gibPerSec(asum_bytes, as_p90),
            as_p99,
            blazt.bench.gibPerSec(asum_bytes, as_p99),
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
            nr_p50,
            blazt.bench.gibPerSec(asum_bytes, nr_p50),
            nr_p90,
            blazt.bench.gibPerSec(asum_bytes, nr_p90),
            nr_p99,
            blazt.bench.gibPerSec(asum_bytes, nr_p99),
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
            imax_p50,
            blazt.bench.gibPerSec(asum_bytes, imax_p50),
            imax_p90,
            blazt.bench.gibPerSec(asum_bytes, imax_p90),
            imax_p99,
            blazt.bench.gibPerSec(asum_bytes, imax_p99),
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
            imin_p50,
            blazt.bench.gibPerSec(asum_bytes, imin_p50),
            imin_p90,
            blazt.bench.gibPerSec(asum_bytes, imin_p90),
            imin_p99,
            blazt.bench.gibPerSec(asum_bytes, imin_p99),
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
            c_p50,
            blazt.bench.gibPerSec(bytes, c_p50),
            c_p90,
            blazt.bench.gibPerSec(bytes, c_p90),
            c_p99,
            blazt.bench.gibPerSec(bytes, c_p99),
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
            s_p50,
            blazt.bench.gibPerSec(scal_bytes, s_p50),
            s_p90,
            blazt.bench.gibPerSec(scal_bytes, s_p90),
            s_p99,
            blazt.bench.gibPerSec(scal_bytes, s_p99),
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
            w_p50,
            blazt.bench.gibPerSec(swap_bytes, w_p50),
            w_p90,
            blazt.bench.gibPerSec(swap_bytes, w_p90),
            w_p99,
            blazt.bench.gibPerSec(swap_bytes, w_p99),
        },
    );
    try out4.flush();

    // GEMM benchmark (row_major, no_trans, beta=0) - sequential vs parallel scaling.
    const m_gemm: usize = 1024;
    const n_gemm: usize = 1024;
    const k_gemm: usize = 1024;

    var a_gemm = try blazt.Matrix(f32, .row_major).init(alloc, m_gemm, k_gemm);
    defer a_gemm.deinit();
    var b_gemm = try blazt.Matrix(f32, .row_major).init(alloc, k_gemm, n_gemm);
    defer b_gemm.deinit();
    var c_gemm = try blazt.Matrix(f32, .row_major).init(alloc, m_gemm, n_gemm);
    defer c_gemm.deinit();

    for (a_gemm.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (b_gemm.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    const gemm_flops: u64 = @intCast(@as(u64, 2) * @as(u64, m_gemm) * @as(u64, n_gemm) * @as(u64, k_gemm));
    const out_gemm = &stdout_writer.interface;

    var res_gemm = try blazt.bench.run(alloc, "ops.gemm(f32,row_major)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, gemmOps, .{ m_gemm, n_gemm, k_gemm, a_gemm, b_gemm, &c_gemm });
    defer res_gemm.deinit();
    res_gemm.sortInPlace();
    const gg_p50 = blazt.bench.medianSortedNs(res_gemm.samples_ns);
    const gg_p90 = blazt.bench.percentileSortedNs(res_gemm.samples_ns, 0.90);
    const gg_p99 = blazt.bench.percentileSortedNs(res_gemm.samples_ns, 0.99);
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_gemm.name,
            gg_p50,
            blazt.bench.gflops(gemm_flops, gg_p50),
            gg_p90,
            blazt.bench.gflops(gemm_flops, gg_p90),
            gg_p99,
            blazt.bench.gflops(gemm_flops, gg_p99),
        },
    );
    try out_gemm.flush();

    // SYRK (rank-k update) bench (row_major, upper, no_trans).
    const n_syrk: usize = 1024;
    const k_syrk: usize = 256;
    var a_syrk = try blazt.Matrix(f32, .row_major).init(alloc, n_syrk, k_syrk);
    defer a_syrk.deinit();
    var c_syrk = try blazt.Matrix(f32, .row_major).init(alloc, n_syrk, n_syrk);
    defer c_syrk.deinit();

    for (a_syrk.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (c_syrk.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_syrk = try blazt.bench.run(alloc, "ops.syrk(f32,row_major,upper)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, syrkOps, .{ n_syrk, k_syrk, a_syrk, &c_syrk });
    defer res_syrk.deinit();
    res_syrk.sortInPlace();
    const syrk_p50 = blazt.bench.medianSortedNs(res_syrk.samples_ns);
    const syrk_p90 = blazt.bench.percentileSortedNs(res_syrk.samples_ns, 0.90);
    const syrk_p99 = blazt.bench.percentileSortedNs(res_syrk.samples_ns, 0.99);

    // SYRK flops: 2*k * (n*(n+1)/2) = k*n*(n+1)
    const syrk_flops: u64 = @intCast(@as(u64, k_syrk) * @as(u64, n_syrk) * @as(u64, n_syrk + 1));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_syrk.name,
            syrk_p50,
            blazt.bench.gflops(syrk_flops, syrk_p50),
            syrk_p90,
            blazt.bench.gflops(syrk_flops, syrk_p90),
            syrk_p99,
            blazt.bench.gflops(syrk_flops, syrk_p99),
        },
    );
    try out_gemm.flush();

    // SYR2K (rank-2k update) bench (row_major, upper, no_trans).
    const n_syr2k: usize = 1024;
    const k_syr2k: usize = 256;
    var a_syr2k = try blazt.Matrix(f32, .row_major).init(alloc, n_syr2k, k_syr2k);
    defer a_syr2k.deinit();
    var b_syr2k = try blazt.Matrix(f32, .row_major).init(alloc, n_syr2k, k_syr2k);
    defer b_syr2k.deinit();
    var c_syr2k = try blazt.Matrix(f32, .row_major).init(alloc, n_syr2k, n_syr2k);
    defer c_syr2k.deinit();

    for (a_syr2k.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 3) % 1024)))) * @as(f32, 0.001);
    for (b_syr2k.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 11) % 1024)))) * @as(f32, 0.0011);
    for (c_syr2k.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_syr2k = try blazt.bench.run(alloc, "ops.syr2k(f32,row_major,upper)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, syr2kOps, .{ n_syr2k, k_syr2k, a_syr2k, b_syr2k, &c_syr2k });
    defer res_syr2k.deinit();
    res_syr2k.sortInPlace();
    const syr2k_p50 = blazt.bench.medianSortedNs(res_syr2k.samples_ns);
    const syr2k_p90 = blazt.bench.percentileSortedNs(res_syr2k.samples_ns, 0.90);
    const syr2k_p99 = blazt.bench.percentileSortedNs(res_syr2k.samples_ns, 0.99);

    // SYR2K flops: 2 terms, each 2*k * (n*(n+1)/2) = 2*k*n*(n+1)
    const syr2k_flops: u64 = @intCast(@as(u64, 2) * @as(u64, k_syr2k) * @as(u64, n_syr2k) * @as(u64, n_syr2k + 1));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_syr2k.name,
            syr2k_p50,
            blazt.bench.gflops(syr2k_flops, syr2k_p50),
            syr2k_p90,
            blazt.bench.gflops(syr2k_flops, syr2k_p90),
            syr2k_p99,
            blazt.bench.gflops(syr2k_flops, syr2k_p99),
        },
    );
    try out_gemm.flush();

    // SYMM bench (row_major, side=left, upper, no_trans).
    const m_symm: usize = 1024;
    const n_symm: usize = 256;
    var a_symm = try blazt.Matrix(f32, .row_major).init(alloc, m_symm, m_symm);
    defer a_symm.deinit();
    var b_symm = try blazt.Matrix(f32, .row_major).init(alloc, m_symm, n_symm);
    defer b_symm.deinit();
    var c_symm = try blazt.Matrix(f32, .row_major).init(alloc, m_symm, n_symm);
    defer c_symm.deinit();

    // Fill only upper triangle for A, and mirror it (so bench isn't dominated by NaN handling).
    for (0..m_symm) |j| {
        for (0..m_symm) |i| {
            if (i <= j) {
                a_symm.atPtr(i, j).* = @as(f32, @floatFromInt(@as(u32, @intCast((i * 37 + j * 17 + 11) % 1024)))) * @as(f32, 0.001);
            } else {
                a_symm.atPtr(i, j).* = a_symm.at(j, i);
            }
        }
    }
    for (b_symm.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 5) % 1024)))) * @as(f32, 0.0013);
    for (c_symm.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_symm = try blazt.bench.run(alloc, "ops.symm(f32,row_major,left,upper)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, symmOps, .{ m_symm, n_symm, a_symm, b_symm, &c_symm });
    defer res_symm.deinit();
    res_symm.sortInPlace();
    const symm_p50 = blazt.bench.medianSortedNs(res_symm.samples_ns);
    const symm_p90 = blazt.bench.percentileSortedNs(res_symm.samples_ns, 0.90);
    const symm_p99 = blazt.bench.percentileSortedNs(res_symm.samples_ns, 0.99);

    // Approx SYMM flops (left): same as GEMM: 2*m*m*n
    const symm_flops: u64 = @intCast(@as(u64, 2) * @as(u64, m_symm) * @as(u64, m_symm) * @as(u64, n_symm));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_symm.name,
            symm_p50,
            blazt.bench.gflops(symm_flops, symm_p50),
            symm_p90,
            blazt.bench.gflops(symm_flops, symm_p90),
            symm_p99,
            blazt.bench.gflops(symm_flops, symm_p99),
        },
    );
    try out_gemm.flush();

    // TRMM/TRSM bench (row_major, side=left, upper, no_trans, non_unit).
    const m_tr: usize = 1024;
    const n_tr: usize = 256;
    var a_tr = try blazt.Matrix(f32, .row_major).init(alloc, m_tr, m_tr);
    defer a_tr.deinit();
    var b0_tr = try blazt.Matrix(f32, .row_major).init(alloc, m_tr, n_tr);
    defer b0_tr.deinit();
    var b_tr = try blazt.Matrix(f32, .row_major).init(alloc, m_tr, n_tr);
    defer b_tr.deinit();

    // Upper triangular A with nonzero diagonal; lower triangle arbitrary.
    for (0..m_tr) |j| {
        for (0..m_tr) |i| {
            if (i <= j) {
                if (i == j) {
                    a_tr.atPtr(i, j).* = 1.5;
                } else {
                    a_tr.atPtr(i, j).* = @as(f32, @floatFromInt(@as(u32, @intCast((i * 37 + j * 17 + 11) % 1024)))) * @as(f32, 0.001);
                }
            } else {
                a_tr.atPtr(i, j).* = 0.0;
            }
        }
    }
    for (b0_tr.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 5) % 1024)))) * @as(f32, 0.0013);
    @memcpy(b_tr.data, b0_tr.data);

    var res_trmm = try blazt.bench.run(alloc, "ops.trmm(f32,row_major,left,upper)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, trmmOps, .{ m_tr, n_tr, a_tr, b0_tr, &b_tr });
    defer res_trmm.deinit();
    res_trmm.sortInPlace();
    const trmm_p50 = blazt.bench.medianSortedNs(res_trmm.samples_ns);
    const trmm_p90 = blazt.bench.percentileSortedNs(res_trmm.samples_ns, 0.90);
    const trmm_p99 = blazt.bench.percentileSortedNs(res_trmm.samples_ns, 0.99);

    // TRMM flops (left): ~ n*m*(m+1)
    const trmm_flops: u64 = @intCast(@as(u64, n_tr) * @as(u64, m_tr) * @as(u64, m_tr + 1));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_trmm.name,
            trmm_p50,
            blazt.bench.gflops(trmm_flops, trmm_p50),
            trmm_p90,
            blazt.bench.gflops(trmm_flops, trmm_p90),
            trmm_p99,
            blazt.bench.gflops(trmm_flops, trmm_p99),
        },
    );
    try out_gemm.flush();

    var res_trsm = try blazt.bench.run(alloc, "ops.trsm(f32,row_major,left,upper)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, trsmOps, .{ m_tr, n_tr, a_tr, b0_tr, &b_tr });
    defer res_trsm.deinit();
    res_trsm.sortInPlace();
    const trsm_p50 = blazt.bench.medianSortedNs(res_trsm.samples_ns);
    const trsm_p90 = blazt.bench.percentileSortedNs(res_trsm.samples_ns, 0.90);
    const trsm_p99 = blazt.bench.percentileSortedNs(res_trsm.samples_ns, 0.99);

    // TRSM flops (left): ~ m*m*n (multiply-add terms) (ignores divides)
    const trsm_flops: u64 = @intCast(@as(u64, m_tr) * @as(u64, m_tr) * @as(u64, n_tr));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_trsm.name,
            trsm_p50,
            blazt.bench.gflops(trsm_flops, trsm_p50),
            trsm_p90,
            blazt.bench.gflops(trsm_flops, trsm_p90),
            trsm_p99,
            blazt.bench.gflops(trsm_flops, trsm_p99),
        },
    );
    try out_gemm.flush();

    // LU bench (row_major, partial pivoting).
    const n_lu: usize = 512;
    var a0_lu = try blazt.Matrix(f32, .row_major).init(alloc, n_lu, n_lu);
    defer a0_lu.deinit();
    var a_lu = try blazt.Matrix(f32, .row_major).init(alloc, n_lu, n_lu);
    defer a_lu.deinit();
    const ipiv_lu = try alloc.alloc(i32, n_lu);
    defer alloc.free(ipiv_lu);

    for (0..n_lu) |i| {
        for (0..n_lu) |j| {
            a0_lu.atPtr(i, j).* = @as(f32, @floatFromInt(@as(u32, @intCast((i * 37 + j * 17 + 11) % 1024)))) * @as(f32, 0.001);
        }
        a0_lu.atPtr(i, i).* += 5.0;
    }

    var res_lu = try blazt.bench.run(alloc, "ops.lu(f32,row_major)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, luOps, .{ n_lu, a0_lu, &a_lu, ipiv_lu });
    defer res_lu.deinit();
    res_lu.sortInPlace();
    const lu_p50 = blazt.bench.medianSortedNs(res_lu.samples_ns);
    const lu_p90 = blazt.bench.percentileSortedNs(res_lu.samples_ns, 0.90);
    const lu_p99 = blazt.bench.percentileSortedNs(res_lu.samples_ns, 0.99);

    // LU flops ~ 2/3 * n^3.
    const lu_flops: u64 = @intCast((@as(u128, 2) * @as(u128, n_lu) * @as(u128, n_lu) * @as(u128, n_lu)) / 3);
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_lu.name,
            lu_p50,
            blazt.bench.gflops(lu_flops, lu_p50),
            lu_p90,
            blazt.bench.gflops(lu_flops, lu_p90),
            lu_p99,
            blazt.bench.gflops(lu_flops, lu_p99),
        },
    );
    try out_gemm.flush();

    // Cholesky bench (row_major, lower).
    const n_chol: usize = 512;
    var a0_chol = try blazt.Matrix(f32, .row_major).init(alloc, n_chol, n_chol);
    defer a0_chol.deinit();
    var a_chol = try blazt.Matrix(f32, .row_major).init(alloc, n_chol, n_chol);
    defer a_chol.deinit();

    // Symmetric strictly diagonally dominant => SPD.
    for (0..n_chol) |i| {
        for (0..n_chol) |j| {
            if (i == j) {
                a0_chol.atPtr(i, j).* = 10.0;
            } else {
                const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 37 + j * 17 + 11) % 1024)))) * @as(f32, 1e-6);
                a0_chol.atPtr(i, j).* = v;
            }
        }
    }
    // Symmetrize explicitly.
    for (0..n_chol) |j| {
        for (j + 1..n_chol) |i| {
            a0_chol.atPtr(i, j).* = a0_chol.at(j, i);
        }
    }

    var res_chol = try blazt.bench.run(alloc, "ops.cholesky(f32,row_major,lower)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, choleskyOps, .{ a0_chol, &a_chol });
    defer res_chol.deinit();
    res_chol.sortInPlace();
    const chol_p50 = blazt.bench.medianSortedNs(res_chol.samples_ns);
    const chol_p90 = blazt.bench.percentileSortedNs(res_chol.samples_ns, 0.90);
    const chol_p99 = blazt.bench.percentileSortedNs(res_chol.samples_ns, 0.99);

    // Cholesky flops ~ 1/3 * n^3.
    const chol_flops: u64 = @intCast((@as(u128, n_chol) * @as(u128, n_chol) * @as(u128, n_chol)) / 3);
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_chol.name,
            chol_p50,
            blazt.bench.gflops(chol_flops, chol_p50),
            chol_p90,
            blazt.bench.gflops(chol_flops, chol_p90),
            chol_p99,
            blazt.bench.gflops(chol_flops, chol_p99),
        },
    );
    try out_gemm.flush();

    // QR bench (row_major, Householder).
    const m_qr: usize = 1024;
    const n_qr: usize = 512;
    const k_qr: usize = @min(m_qr, n_qr);
    var a0_qr = try blazt.Matrix(f32, .row_major).init(alloc, m_qr, n_qr);
    defer a0_qr.deinit();
    var a_qr = try blazt.Matrix(f32, .row_major).init(alloc, m_qr, n_qr);
    defer a_qr.deinit();
    const tau_qr = try alloc.alloc(f32, k_qr);
    defer alloc.free(tau_qr);

    for (a0_qr.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i * 37 + 11) % 1024)))) * @as(f32, 0.001);

    var res_qr = try blazt.bench.run(alloc, "ops.qr(f32,row_major)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, qrOps, .{ m_qr, n_qr, a0_qr, &a_qr, tau_qr });
    defer res_qr.deinit();
    res_qr.sortInPlace();
    const qr_p50 = blazt.bench.medianSortedNs(res_qr.samples_ns);
    const qr_p90 = blazt.bench.percentileSortedNs(res_qr.samples_ns, 0.90);
    const qr_p99 = blazt.bench.percentileSortedNs(res_qr.samples_ns, 0.99);

    // QR flops (m>=n): ~ 2*m*n^2 - 2/3*n^3
    const qr_flops: u64 = @intCast((@as(u128, 2) * @as(u128, m_qr) * @as(u128, n_qr) * @as(u128, n_qr)) -
        ((@as(u128, 2) * @as(u128, n_qr) * @as(u128, n_qr) * @as(u128, n_qr)) / 3));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_qr.name,
            qr_p50,
            blazt.bench.gflops(qr_flops, qr_p50),
            qr_p90,
            blazt.bench.gflops(qr_flops, qr_p90),
            qr_p99,
            blazt.bench.gflops(qr_flops, qr_p99),
        },
    );
    try out_gemm.flush();

    // SVD bench (row_major, economy, f32). (Jacobi; this is expensive.)
    const m_svd: usize = if (lapack_heavy) 256 else 96;
    const n_svd: usize = if (lapack_heavy) 128 else 48;
    const k_svd: usize = @min(m_svd, n_svd);
    var a0_svd = try blazt.Matrix(f32, .row_major).init(alloc, m_svd, n_svd);
    defer a0_svd.deinit();
    var a_svd = try blazt.Matrix(f32, .row_major).init(alloc, m_svd, n_svd);
    defer a_svd.deinit();
    var u_svd = try blazt.Matrix(f32, .row_major).init(alloc, m_svd, k_svd);
    defer u_svd.deinit();
    var vt_svd = try blazt.Matrix(f32, .row_major).init(alloc, k_svd, n_svd);
    defer vt_svd.deinit();
    const s_svd = try alloc.alloc(f32, k_svd);
    defer alloc.free(s_svd);

    for (a0_svd.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i * 17 + 3) % 1024)))) * @as(f32, 0.001);

    if (lapack_heavy) {
        std.debug.print("running ops.svd(f32,row_major) (heavy; set BLAZT_BENCH_LAPACK_HEAVY=0 to skip heavy sizes)\n", .{});
    }
    var res_svd = try blazt.bench.run(alloc, "ops.svd(f32,row_major)", .{
        .warmup_iters = if (lapack_heavy) 1 else 0,
        .samples = if (lapack_heavy) 5 else 1,
        .inner_iters = 1,
    }, svdOps, .{ m_svd, n_svd, a0_svd, &a_svd, s_svd, &u_svd, &vt_svd });
    defer res_svd.deinit();
    res_svd.sortInPlace();
    const svd_p50 = blazt.bench.medianSortedNs(res_svd.samples_ns);
    const svd_p90 = blazt.bench.percentileSortedNs(res_svd.samples_ns, 0.90);
    const svd_p99 = blazt.bench.percentileSortedNs(res_svd.samples_ns, 0.99);

    // Coarse SVD flops estimate: O(m*n^2) for m>=n.
    const svd_flops: u64 = @intCast(@as(u128, 8) * @as(u128, m_svd) * @as(u128, n_svd) * @as(u128, n_svd));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_svd.name,
            svd_p50,
            blazt.bench.gflops(svd_flops, svd_p50),
            svd_p90,
            blazt.bench.gflops(svd_flops, svd_p90),
            svd_p99,
            blazt.bench.gflops(svd_flops, svd_p99),
        },
    );
    try out_gemm.flush();

    // Eig bench (symmetric, row_major, f32). (Jacobi; this is expensive.)
    const n_eig: usize = if (lapack_heavy) 128 else 64;
    var a0_eig = try blazt.Matrix(f32, .row_major).init(alloc, n_eig, n_eig);
    defer a0_eig.deinit();
    var a_eig = try blazt.Matrix(f32, .row_major).init(alloc, n_eig, n_eig);
    defer a_eig.deinit();
    var v_eig = try blazt.Matrix(f32, .row_major).init(alloc, n_eig, n_eig);
    defer v_eig.deinit();
    const w_eig = try alloc.alloc(f32, n_eig);
    defer alloc.free(w_eig);

    // Build a symmetric matrix.
    for (0..n_eig) |i| {
        for (0..n_eig) |j| {
            const base = @as(f32, @floatFromInt(@as(u32, @intCast(((i * 37 + j * 17 + 11) ^ 0x5bd1e995) % 1024)))) * @as(f32, 0.001);
            if (i <= j) {
                a0_eig.atPtr(i, j).* = base;
            } else {
                a0_eig.atPtr(i, j).* = a0_eig.at(j, i);
            }
        }
    }

    if (lapack_heavy) {
        std.debug.print("running ops.eig(f32,row_major,symmetric) (heavy; set BLAZT_BENCH_LAPACK_HEAVY=0 to skip heavy sizes)\n", .{});
    }
    var res_eig = try blazt.bench.run(alloc, "ops.eig(f32,row_major,symmetric)", .{
        .warmup_iters = if (lapack_heavy) 1 else 0,
        .samples = if (lapack_heavy) 5 else 1,
        .inner_iters = 1,
    }, eigOps, .{ n_eig, a0_eig, &a_eig, w_eig, &v_eig });
    defer res_eig.deinit();
    res_eig.sortInPlace();
    const eig_p50 = blazt.bench.medianSortedNs(res_eig.samples_ns);
    const eig_p90 = blazt.bench.percentileSortedNs(res_eig.samples_ns, 0.90);
    const eig_p99 = blazt.bench.percentileSortedNs(res_eig.samples_ns, 0.99);

    const eig_flops: u64 = @intCast(@as(u128, 10) * @as(u128, n_eig) * @as(u128, n_eig) * @as(u128, n_eig));
    try out_gemm.print(
        "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
        .{
            res_eig.name,
            eig_p50,
            blazt.bench.gflops(eig_flops, eig_p50),
            eig_p90,
            blazt.bench.gflops(eig_flops, eig_p90),
            eig_p99,
            blazt.bench.gflops(eig_flops, eig_p99),
        },
    );
    try out_gemm.flush();

    // Parallel variants (fixed-size pools so we don't include init costs).
    //
    // Scaling story points (hyperthreading span): 8,12,16,24,28,32. Keep 2/4 for baseline.
    const cpu_threads: usize = std.Thread.getCpuCount() catch 8;
    const max_threads: usize = @min(@as(usize, 32), @max(cpu_threads, 1));

    const desired_threads = [_]usize{ 2, 4, 8, 12, 16, 24, 28, 32 };
    var last_threads: usize = 0;
    for (desired_threads) |tc| {
        const threads: usize = if (tc == 32) max_threads else tc;
        if (threads <= 1) continue;
        if (threads > cpu_threads) continue;
        if (threads == last_threads) continue;
        last_threads = threads;

        var pool = try blazt.ThreadPool.init(alloc, .{
            .thread_count = threads,
            .pin_threads = true,
        });
        defer pool.deinit();

        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "parallel.gemm(f32,row_major,threads={d})", .{threads});

        var res_pg = try blazt.bench.run(alloc, name, .{
            .warmup_iters = 2,
            .samples = 10,
            .inner_iters = 1,
        }, gemmParallelOps, .{ m_gemm, n_gemm, k_gemm, a_gemm, b_gemm, &c_gemm, &pool });
        defer res_pg.deinit();
        res_pg.sortInPlace();
        const pg_p50 = blazt.bench.medianSortedNs(res_pg.samples_ns);
        const pg_p90 = blazt.bench.percentileSortedNs(res_pg.samples_ns, 0.90);
        const pg_p99 = blazt.bench.percentileSortedNs(res_pg.samples_ns, 0.99);
        try out_gemm.print(
            "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
            .{
                res_pg.name,
                pg_p50,
                blazt.bench.gflops(gemm_flops, pg_p50),
                pg_p90,
                blazt.bench.gflops(gemm_flops, pg_p90),
                pg_p99,
                blazt.bench.gflops(gemm_flops, pg_p99),
            },
        );
        try out_gemm.flush();
    }

    // %peak reporting:
    // - set `BLAZT_PEAK_GFLOPS` to your CPU's theoretical peak GFLOP/s
    // - %peak = (measured_gflops / peak_gflops) * 100
    const peak_gflops: ?f64 = blk: {
        const s = std.process.getEnvVarOwned(alloc, "BLAZT_PEAK_GFLOPS") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => break :blk null,
            else => return err,
        };
        defer alloc.free(s);
        break :blk std.fmt.parseFloat(f64, s) catch null;
    };

    // Oracle comparison (col_major) when available.
    const do_oracle_bench: bool = blk: {
        const s = std.process.getEnvVarOwned(alloc, "BLAZT_BENCH_ORACLE") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => break :blk false,
            else => return err,
        };
        defer alloc.free(s);
        // Anything other than "0" enables it (e.g. "1", "true").
        break :blk s.len != 0 and !std.mem.eql(u8, s, "0");
    };

    // NOTE: We intentionally load+bench+unload each oracle one-at-a-time.
    // Some BLAS libraries create thread pools at load time (or first call) that can remain
    // active and interfere with subsequent benchmarks of other libraries.
    const oracle_kinds = [_]blazt.oracle.Oracle.Kind{ .openblas, .blis, .mkl };

    var a_cm = try blazt.Matrix(f32, .col_major).init(alloc, m_gemm, k_gemm);
    defer a_cm.deinit();
    var b_cm = try blazt.Matrix(f32, .col_major).init(alloc, k_gemm, n_gemm);
    defer b_cm.deinit();
    var c_cm = try blazt.Matrix(f32, .col_major).init(alloc, m_gemm, n_gemm);
    defer c_cm.deinit();

    for (a_cm.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
    for (b_cm.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

    var res_gemm_cm = try blazt.bench.run(alloc, "ops.gemm(f32,col_major)", .{
        .warmup_iters = 2,
        .samples = 10,
        .inner_iters = 1,
    }, gemmOpsColMajor, .{ m_gemm, n_gemm, k_gemm, a_cm, b_cm, &c_cm });
    defer res_gemm_cm.deinit();
    res_gemm_cm.sortInPlace();
    const cm_p50 = blazt.bench.medianSortedNs(res_gemm_cm.samples_ns);
    const cm_p90 = blazt.bench.percentileSortedNs(res_gemm_cm.samples_ns, 0.90);
    const cm_p99 = blazt.bench.percentileSortedNs(res_gemm_cm.samples_ns, 0.99);

    if (peak_gflops) |peak| {
        const p50_g = blazt.bench.gflops(gemm_flops, cm_p50);
        const pct = (p50_g / peak) * 100.0;
        try out_gemm.print(
            "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s, {d:.1}% peak)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
            .{
                res_gemm_cm.name,
                cm_p50,
                p50_g,
                pct,
                cm_p90,
                blazt.bench.gflops(gemm_flops, cm_p90),
                cm_p99,
                blazt.bench.gflops(gemm_flops, cm_p99),
            },
        );
    } else {
        try out_gemm.print(
            "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
            .{
                res_gemm_cm.name,
                cm_p50,
                blazt.bench.gflops(gemm_flops, cm_p50),
                cm_p90,
                blazt.bench.gflops(gemm_flops, cm_p90),
                cm_p99,
                blazt.bench.gflops(gemm_flops, cm_p99),
            },
        );
    }
    try out_gemm.flush();

    if (do_oracle_bench) {
        // Oracle comparison uses the same size as the main GEMM benchmark so the numbers
        // are directly comparable.
        const m_or: usize = m_gemm;
        const n_or: usize = n_gemm;
        const k_or: usize = k_gemm;
        const flops_or: u64 = gemm_flops;

        var a_or = try blazt.Matrix(f32, .col_major).init(alloc, m_or, k_or);
        defer a_or.deinit();
        var b_or = try blazt.Matrix(f32, .col_major).init(alloc, k_or, n_or);
        defer b_or.deinit();
        var c_or = try blazt.Matrix(f32, .col_major).init(alloc, m_or, n_or);
        defer c_or.deinit();

        for (a_or.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
        for (b_or.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);

        var res_gemm_or = try blazt.bench.run(alloc, "ops.gemm(f32,col_major,oracle_cmp)", .{
            .warmup_iters = 2,
            .samples = 10,
            .inner_iters = 1,
        }, gemmOpsColMajor, .{ m_or, n_or, k_or, a_or, b_or, &c_or });
        defer res_gemm_or.deinit();
        res_gemm_or.sortInPlace();
        const g_or_p50 = blazt.bench.medianSortedNs(res_gemm_or.samples_ns);
        const g_or_p90 = blazt.bench.percentileSortedNs(res_gemm_or.samples_ns, 0.90);
        const g_or_p99 = blazt.bench.percentileSortedNs(res_gemm_or.samples_ns, 0.99);
        try out_gemm.print(
            "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
            .{
                res_gemm_or.name,
                g_or_p50,
                blazt.bench.gflops(flops_or, g_or_p50),
                g_or_p90,
                blazt.bench.gflops(flops_or, g_or_p90),
                g_or_p99,
                blazt.bench.gflops(flops_or, g_or_p99),
            },
        );
        try out_gemm.flush();

        for (oracle_kinds) |kind| {
            var o = blazt.oracle.Oracle.loadKind(alloc, kind) catch |err| switch (err) {
                error.LibraryNotFound => continue,
                else => return err,
            };
            defer o.unload();
            const oracle_name = switch (o.kind) {
                .openblas => "oracle.sgemm(openblas,col_major)",
                .blis => "oracle.sgemm(blis,col_major)",
                .mkl => "oracle.sgemm(mkl,col_major)",
            };
            var res_or = try blazt.bench.run(alloc, oracle_name, .{
                .warmup_iters = 2,
                .samples = 10,
                .inner_iters = 1,
            }, oracleSgemmOps, .{ &o, m_or, n_or, k_or, a_or, b_or, &c_or });
            defer res_or.deinit();
            res_or.sortInPlace();
            const or_p50 = blazt.bench.medianSortedNs(res_or.samples_ns);
            const or_p90 = blazt.bench.percentileSortedNs(res_or.samples_ns, 0.90);
            const or_p99 = blazt.bench.percentileSortedNs(res_or.samples_ns, 0.99);
            try out_gemm.print(
                "bench {s}\n  p50: {d} ns  ({d:.3} GFLOP/s)\n  p90: {d} ns  ({d:.3} GFLOP/s)\n  p99: {d} ns  ({d:.3} GFLOP/s)\n",
                .{
                    res_or.name,
                    or_p50,
                    blazt.bench.gflops(flops_or, or_p50),
                    or_p90,
                    blazt.bench.gflops(flops_or, or_p90),
                    or_p99,
                    blazt.bench.gflops(flops_or, or_p99),
                },
            );
            try out_gemm.flush();
        }
    }

    try benchCacheHotAfterZero(alloc, out_gemm);
    try out_gemm.flush();
}
