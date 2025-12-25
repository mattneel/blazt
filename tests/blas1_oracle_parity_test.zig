const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn lookupFn(lib: *std.DynLib, comptime Fn: type, name: [:0]const u8) ?Fn {
    const sym = lib.lookup(*anyopaque, name) orelse return null;
    return @as(Fn, @ptrFromInt(@intFromPtr(sym)));
}

const F77ScopyFn = *const fn (n: *const c_int, x: [*]const f32, incx: *const c_int, y: [*]f32, incy: *const c_int) callconv(.c) void;
const F77SscalFn = *const fn (n: *const c_int, alpha: *const f32, x: [*]f32, incx: *const c_int) callconv(.c) void;
const F77SswapFn = *const fn (n: *const c_int, x: [*]f32, incx: *const c_int, y: [*]f32, incy: *const c_int) callconv(.c) void;
const F77SaxpyFn = *const fn (n: *const c_int, alpha: *const f32, x: [*]const f32, incx: *const c_int, y: [*]f32, incy: *const c_int) callconv(.c) void;
const F77SdotFn = *const fn (n: *const c_int, x: [*]const f32, incx: *const c_int, y: [*]const f32, incy: *const c_int) callconv(.c) f32;
const F77Snrm2Fn = *const fn (n: *const c_int, x: [*]const f32, incx: *const c_int) callconv(.c) f32;
const F77SasumFn = *const fn (n: *const c_int, x: [*]const f32, incx: *const c_int) callconv(.c) f32;
const F77IsamaxFn = *const fn (n: *const c_int, x: [*]const f32, incx: *const c_int) callconv(.c) c_int;
const F77IsaminFn = *const fn (n: *const c_int, x: [*]const f32, incx: *const c_int) callconv(.c) c_int;

test "BLAS1 parity vs oracle (when available)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const lib = &oracle.lib;

    const scopy = lookupFn(lib, F77ScopyFn, "scopy_") orelse return error.SkipZigTest;
    const sscal = lookupFn(lib, F77SscalFn, "sscal_") orelse return error.SkipZigTest;
    const sswap = lookupFn(lib, F77SswapFn, "sswap_") orelse return error.SkipZigTest;
    const saxpy = lookupFn(lib, F77SaxpyFn, "saxpy_") orelse return error.SkipZigTest;
    const sdot = lookupFn(lib, F77SdotFn, "sdot_") orelse return error.SkipZigTest;
    const snrm2 = lookupFn(lib, F77Snrm2Fn, "snrm2_") orelse return error.SkipZigTest;
    const sasum = lookupFn(lib, F77SasumFn, "sasum_") orelse return error.SkipZigTest;
    const isamax = lookupFn(lib, F77IsamaxFn, "isamax_") orelse return error.SkipZigTest;
    const isamin = lookupFn(lib, F77IsaminFn, "isamin_") orelse return error.SkipZigTest;

    // Parity tolerance:
    // - For elementwise ops (copy/scal/swap/axpy), results should match closely.
    // - For reductions (dot/nrm2/asum), different accumulation strategies across BLAS impls can differ by many ulps.
    const tol_elem = fx.defaultTolerance(f32);
    const tol_reduce = fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 };

    const n: usize = 1024 + 3; // odd size to hit tails
    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);

    // Deterministic init (no NaNs).
    for (x, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * @as(f32, 0.001);
        v.* = if ((i & 1) == 0) base else -base;
    }
    for (y, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * @as(f32, 0.002);
    }

    const one: c_int = 1;
    const n_i32: c_int = @intCast(n);

    // copy
    const y_blazt = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(y_blazt);
    const y_oracle = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(y_oracle);

    blazt.ops.copy(f32, n, x, y_blazt);
    scopy(&n_i32, x.ptr, &one, y_oracle.ptr, &one);
    try fx.expectSliceApproxEq(f32, y_oracle, y_blazt, tol_elem);

    // scal
    const alpha: f32 = 0.5;
    const x_blazt = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(x_blazt);
    const x_oracle = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(x_oracle);

    blazt.ops.scal(f32, n, alpha, x_blazt);
    var alpha_v = alpha;
    sscal(&n_i32, &alpha_v, x_oracle.ptr, &one);
    try fx.expectSliceApproxEq(f32, x_oracle, x_blazt, tol_elem);

    // swap
    const sx_blazt = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(sx_blazt);
    const sy_blazt = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(sy_blazt);
    const sx_oracle = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(sx_oracle);
    const sy_oracle = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(sy_oracle);

    blazt.ops.swap(f32, n, sx_blazt, sy_blazt);
    sswap(&n_i32, sx_oracle.ptr, &one, sy_oracle.ptr, &one);
    try fx.expectSliceApproxEq(f32, sx_oracle, sx_blazt, tol_elem);
    try fx.expectSliceApproxEq(f32, sy_oracle, sy_blazt, tol_elem);

    // axpy
    const ay_blazt = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(ay_blazt);
    const ay_oracle = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(ay_oracle);

    blazt.ops.axpy(f32, n, alpha, x, ay_blazt);
    saxpy(&n_i32, &alpha_v, x.ptr, &one, ay_oracle.ptr, &one);
    try fx.expectSliceApproxEq(f32, ay_oracle, ay_blazt, tol_elem);

    // dot / nrm2 / asum
    const dot_blazt = blazt.ops.dot(f32, x, y);
    const dot_oracle = sdot(&n_i32, x.ptr, &one, y.ptr, &one);
    try fx.expectFloatApproxEq(f32, dot_oracle, dot_blazt, tol_reduce);

    const nrm2_blazt = blazt.ops.nrm2(f32, x);
    const nrm2_oracle = snrm2(&n_i32, x.ptr, &one);
    try fx.expectFloatApproxEq(f32, nrm2_oracle, nrm2_blazt, tol_reduce);

    const asum_blazt = blazt.ops.asum(f32, x);
    const asum_oracle = sasum(&n_i32, x.ptr, &one);
    try fx.expectFloatApproxEq(f32, asum_oracle, asum_blazt, tol_reduce);

    // iamax / iamin (oracle returns 1-based index; 0 means n < 1)
    const imax_oracle = isamax(&n_i32, x.ptr, &one);
    const imin_oracle = isamin(&n_i32, x.ptr, &one);
    if (imax_oracle <= 0 or imin_oracle <= 0) return error.TestUnexpectedError;

    const imax_blazt = blazt.ops.iamax(f32, x) orelse return error.TestUnexpectedError;
    const imin_blazt = blazt.ops.iamin(f32, x) orelse return error.TestUnexpectedError;

    try std.testing.expectEqual(@as(usize, @intCast(imax_oracle - 1)), imax_blazt);
    try std.testing.expectEqual(@as(usize, @intCast(imin_oracle - 1)), imin_blazt);
}


