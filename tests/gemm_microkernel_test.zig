const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn refMicroKernelF64(
    comptime MR: usize,
    comptime NR: usize,
    kc: usize,
    a: []const f32,
    b: []const f32,
    alpha: f32,
    beta: f32,
    c_in: []const f32,
    c_out: []f32,
    rs_c: usize,
    cs_c: usize,
) void {
    for (0..NR) |j| {
        for (0..MR) |i| {
            var sum: f64 = 0.0;
            for (0..kc) |p| {
                const av = @as(f64, @floatCast(a[p * MR + i]));
                const bv = @as(f64, @floatCast(b[p * NR + j]));
                sum += av * bv;
            }

            const idx = i * rs_c + j * cs_c;
            // beta=0 path must not read C (NaN-poisoned tests rely on this).
            const cin: f64 = if (beta == @as(f32, 0)) 0.0 else @as(f64, @floatCast(c_in[idx]));
            const out = @as(f64, @floatCast(alpha)) * sum + @as(f64, @floatCast(beta)) * cin;
            c_out[idx] = @floatCast(out);
        }
    }
}

test "gemm.microKernel matches reference (beta=0 avoids reading C)" {
    const MR: usize = 4;
    const NR: usize = 3;
    const kc: usize = 7;

    var a: [MR * kc]f32 = undefined;
    var b: [kc * NR]f32 = undefined;

    for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 31)))) * 0.01;
    for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 7) % 29)))) * 0.02;

    const alpha: f32 = 0.5;
    const beta: f32 = 0.0;

    // Column-major C block: idx = i + j*MR.
    const rs_c: usize = 1;
    const cs_c: usize = MR;

    const nan = std.math.nan(f32);
    var c: [MR * NR]f32 = .{nan} ** (MR * NR);

    // Reference uses c_in, but for beta=0 it should not matter; keep NaNs to catch reads.
    var cref: [MR * NR]f32 = undefined;
    refMicroKernelF64(MR, NR, kc, &a, &b, alpha, beta, &c, &cref, rs_c, cs_c);

    blazt.gemm.microKernel(f32, MR, NR, kc, &a, &b, &c, rs_c, cs_c, alpha, beta);

    // Ensure no NaNs were produced (beta=0 path should not read NaN inputs).
    for (c) |v| try std.testing.expect(!std.math.isNan(v));

    try fx.expectSliceApproxEq(f32, &cref, &c, fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 });
}

test "gemm.microKernel matches reference (beta=1 accumulates)" {
    const MR: usize = 4;
    const NR: usize = 4;
    const kc: usize = 5;

    var a: [MR * kc]f32 = undefined;
    var b: [kc * NR]f32 = undefined;

    for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 1) % 23)))) * 0.01;
    for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 3) % 19)))) * 0.02;

    const alpha: f32 = 0.5;
    const beta: f32 = 1.0;

    // Row-major C block: idx = i*NR + j.
    const rs_c: usize = NR;
    const cs_c: usize = 1;

    var c: [MR * NR]f32 = undefined;
    for (&c, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 17)))) * 0.001;

    var cref = c;
    refMicroKernelF64(MR, NR, kc, &a, &b, alpha, beta, &c, &cref, rs_c, cs_c);

    blazt.gemm.microKernel(f32, MR, NR, kc, &a, &b, &c, rs_c, cs_c, alpha, beta);

    try fx.expectSliceApproxEq(f32, &cref, &c, fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 });
}


