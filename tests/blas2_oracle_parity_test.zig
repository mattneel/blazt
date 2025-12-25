const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn lookupFn(lib: *std.DynLib, comptime Fn: type, name: [:0]const u8) ?Fn {
    const sym = lib.lookup(*anyopaque, name) orelse return null;
    return @as(Fn, @ptrFromInt(@intFromPtr(sym)));
}

const F77SgemvFn = *const fn (
    trans: *const u8,
    m: *const c_int,
    n: *const c_int,
    alpha: *const f32,
    a: [*]const f32,
    lda: *const c_int,
    x: [*]const f32,
    incx: *const c_int,
    beta: *const f32,
    y: [*]f32,
    incy: *const c_int,
) callconv(.c) void;

const F77SgerFn = *const fn (
    m: *const c_int,
    n: *const c_int,
    alpha: *const f32,
    x: [*]const f32,
    incx: *const c_int,
    y: [*]const f32,
    incy: *const c_int,
    a: [*]f32,
    lda: *const c_int,
) callconv(.c) void;

const F77StrmvFn = *const fn (
    uplo: *const u8,
    trans: *const u8,
    diag: *const u8,
    n: *const c_int,
    a: [*]const f32,
    lda: *const c_int,
    x: [*]f32,
    incx: *const c_int,
) callconv(.c) void;

const F77StrsvFn = *const fn (
    uplo: *const u8,
    trans: *const u8,
    diag: *const u8,
    n: *const c_int,
    a: [*]const f32,
    lda: *const c_int,
    x: [*]f32,
    incx: *const c_int,
) callconv(.c) void;

const F77SsymvFn = *const fn (
    uplo: *const u8,
    n: *const c_int,
    alpha: *const f32,
    a: [*]const f32,
    lda: *const c_int,
    x: [*]const f32,
    incx: *const c_int,
    beta: *const f32,
    y: [*]f32,
    incy: *const c_int,
) callconv(.c) void;

const ComplexF32 = std.math.Complex(f32);

const F77ChemvFn = *const fn (
    uplo: *const u8,
    n: *const c_int,
    alpha: *const ComplexF32,
    a: [*]const ComplexF32,
    lda: *const c_int,
    x: [*]const ComplexF32,
    incx: *const c_int,
    beta: *const ComplexF32,
    y: [*]ComplexF32,
    incy: *const c_int,
) callconv(.c) void;

fn expectComplexApproxEq(a: ComplexF32, b: ComplexF32, tol: fx.FloatTolerance) !void {
    try fx.expectFloatApproxEq(f32, a.re, b.re, tol);
    try fx.expectFloatApproxEq(f32, a.im, b.im, tol);
}

fn expectComplexSliceApproxEq(a: []const ComplexF32, b: []const ComplexF32, tol: fx.FloatTolerance) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (a, b) |av, bv| try expectComplexApproxEq(av, bv, tol);
}

test "BLAS2 parity: gemv (col_major, no_trans + trans)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const sgemv = lookupFn(&oracle.lib, F77SgemvFn, "sgemv_") orelse return error.SkipZigTest;
    const tol = fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 };

    const one: c_int = 1;
    const m: usize = 7;
    const n: usize = 5;
    const m_i: c_int = @intCast(m);
    const n_i: c_int = @intCast(n);

    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer a.deinit();
    for (a.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * 0.001;

    const x_n = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x_n);
    const y_m = try std.testing.allocator.alloc(f32, m);
    defer std.testing.allocator.free(y_m);

    for (x_n, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 7) % 1024)))) * 0.002;
    for (y_m, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 11) % 1024)))) * 0.003;

    // no_trans: y := alpha*A*x + beta*y
    const alpha: f32 = 0.5;
    const beta: f32 = 0.25;

    const y_blazt = try std.testing.allocator.dupe(f32, y_m);
    defer std.testing.allocator.free(y_blazt);
    const y_oracle = try std.testing.allocator.dupe(f32, y_m);
    defer std.testing.allocator.free(y_oracle);

    blazt.ops.gemv(f32, .col_major, .no_trans, m, n, alpha, a, x_n, beta, y_blazt);

    const trans_n: u8 = 'N';
    var alpha_v = alpha;
    var beta_v = beta;
    var lda: c_int = @intCast(a.stride);
    sgemv(&trans_n, &m_i, &n_i, &alpha_v, a.data.ptr, &lda, x_n.ptr, &one, &beta_v, y_oracle.ptr, &one);

    try fx.expectSliceApproxEq(f32, y_oracle, y_blazt, tol);

    // trans: y := alpha*A^T*x + beta*y
    const x_m = try std.testing.allocator.alloc(f32, m);
    defer std.testing.allocator.free(x_m);
    const y_n = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y_n);

    for (x_m, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 13) % 1024)))) * 0.001;
    for (y_n, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 17) % 1024)))) * 0.002;

    const y2_blazt = try std.testing.allocator.dupe(f32, y_n);
    defer std.testing.allocator.free(y2_blazt);
    const y2_oracle = try std.testing.allocator.dupe(f32, y_n);
    defer std.testing.allocator.free(y2_oracle);

    blazt.ops.gemv(f32, .col_major, .trans, m, n, alpha, a, x_m, beta, y2_blazt);

    const trans_t: u8 = 'T';
    sgemv(&trans_t, &m_i, &n_i, &alpha_v, a.data.ptr, &lda, x_m.ptr, &one, &beta_v, y2_oracle.ptr, &one);

    try fx.expectSliceApproxEq(f32, y2_oracle, y2_blazt, tol);
}

test "BLAS2 parity: ger (col_major)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const sger = lookupFn(&oracle.lib, F77SgerFn, "sger_") orelse return error.SkipZigTest;
    const tol_elem = fx.defaultTolerance(f32);

    const one: c_int = 1;
    const m: usize = 4;
    const n: usize = 3;
    const m_i: c_int = @intCast(m);
    const n_i: c_int = @intCast(n);

    var a0 = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer a0.deinit();
    for (a0.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i % 1024)))) * 0.001;

    const x = try std.testing.allocator.alloc(f32, m);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 7) % 1024)))) * 0.002;
    for (y, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 11) % 1024)))) * 0.003;

    var a_blazt = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer a_blazt.deinit();
    var a_oracle = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, m, n);
    defer a_oracle.deinit();

    @memcpy(a_blazt.data, a0.data);
    @memcpy(a_oracle.data, a0.data);

    const alpha: f32 = 0.5;
    blazt.ops.ger(f32, .col_major, m, n, alpha, x, y, &a_blazt);

    var alpha_v = alpha;
    var lda: c_int = @intCast(a_oracle.stride);
    sger(&m_i, &n_i, &alpha_v, x.ptr, &one, y.ptr, &one, a_oracle.data.ptr, &lda);

    try fx.expectSliceApproxEq(f32, a_oracle.data, a_blazt.data, tol_elem);
}

test "BLAS2 parity: trmv + trsv (col_major, upper/lower + unit/non-unit)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const strmv = lookupFn(&oracle.lib, F77StrmvFn, "strmv_") orelse return error.SkipZigTest;
    const strsv = lookupFn(&oracle.lib, F77StrsvFn, "strsv_") orelse return error.SkipZigTest;
    const tol = fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 };

    const one: c_int = 1;
    const n: usize = 7;
    const n_i: c_int = @intCast(n);

    const nan = std.math.nan(f32);

    // Upper, non-unit, no_trans
    {
        var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memset(a.data, nan);
        // Fill upper triangle with deterministic values; diagonal non-zero.
        for (0..n) |j| {
            for (0..j + 1) |i| {
                const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 17 + j * 31 + 1) % 1024)))) * 0.001;
                a.data[j * a.stride + i] = v;
            }
        }

        const x0 = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(x0);
        for (x0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 5) % 1024)))) * 0.002;

        const xb = try std.testing.allocator.dupe(f32, x0);
        defer std.testing.allocator.free(xb);
        const xo = try std.testing.allocator.dupe(f32, x0);
        defer std.testing.allocator.free(xo);

        blazt.ops.trmv(f32, .col_major, .upper, .no_trans, .non_unit, n, a, xb);

        const uplo: u8 = 'U';
        const trans: u8 = 'N';
        const diag: u8 = 'N';
        var lda: c_int = @intCast(a.stride);
        strmv(&uplo, &trans, &diag, &n_i, a.data.ptr, &lda, xo.ptr, &one);

        try fx.expectSliceApproxEq(f32, xo, xb, tol);

        // Solve the same upper system: use b = A*x0 (computed by trmv above) and recover x0.
        const b = xb;
        const x_solve_blazt = try std.testing.allocator.dupe(f32, b);
        defer std.testing.allocator.free(x_solve_blazt);
        const x_solve_oracle = try std.testing.allocator.dupe(f32, b);
        defer std.testing.allocator.free(x_solve_oracle);

        try blazt.ops.trsv(f32, .col_major, .upper, .no_trans, .non_unit, n, a, x_solve_blazt);
        strsv(&uplo, &trans, &diag, &n_i, a.data.ptr, &lda, x_solve_oracle.ptr, &one);

        try fx.expectSliceApproxEq(f32, x_solve_oracle, x_solve_blazt, tol);
    }

    // Lower, unit, trans (poison diagonal + unused triangle)
    {
        var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memset(a.data, nan);
        // Fill lower triangle below diag; diagonal poisoned (unit diag must ignore).
        for (0..n) |j| {
            for (j..n) |i| {
                if (i == j) {
                    a.data[j * a.stride + i] = nan;
                } else {
                    const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 19 + j * 23 + 3) % 1024)))) * 0.001;
                    a.data[j * a.stride + i] = v;
                }
            }
        }

        const x0 = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(x0);
        for (x0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 9) % 1024)))) * 0.001;

        const xb = try std.testing.allocator.dupe(f32, x0);
        defer std.testing.allocator.free(xb);
        const xo = try std.testing.allocator.dupe(f32, x0);
        defer std.testing.allocator.free(xo);

        blazt.ops.trmv(f32, .col_major, .lower, .trans, .unit, n, a, xb);

        const uplo: u8 = 'L';
        const trans: u8 = 'T';
        const diag: u8 = 'U';
        var lda: c_int = @intCast(a.stride);
        strmv(&uplo, &trans, &diag, &n_i, a.data.ptr, &lda, xo.ptr, &one);

        try fx.expectSliceApproxEq(f32, xo, xb, tol);

        // trsv parity on the same configuration
        const b = xb;
        const x_solve_blazt = try std.testing.allocator.dupe(f32, b);
        defer std.testing.allocator.free(x_solve_blazt);
        const x_solve_oracle = try std.testing.allocator.dupe(f32, b);
        defer std.testing.allocator.free(x_solve_oracle);

        try blazt.ops.trsv(f32, .col_major, .lower, .trans, .unit, n, a, x_solve_blazt);
        strsv(&uplo, &trans, &diag, &n_i, a.data.ptr, &lda, x_solve_oracle.ptr, &one);

        try fx.expectSliceApproxEq(f32, x_solve_oracle, x_solve_blazt, tol);
    }
}

test "BLAS2 parity: symv (col_major, upper/lower)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const ssymv = lookupFn(&oracle.lib, F77SsymvFn, "ssymv_") orelse return error.SkipZigTest;
    const tol = fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 };

    const one: c_int = 1;
    const n: usize = 9;
    const n_i: c_int = @intCast(n);

    const alpha: f32 = 0.5;
    const beta: f32 = 0.25;
    var alpha_v = alpha;
    var beta_v = beta;

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 3) % 1024)))) * 0.001;

    const y0 = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y0);
    for (y0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast((i + 7) % 1024)))) * 0.002;

    const nan = std.math.nan(f32);

    // upper
    {
        var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memset(a.data, nan);
        for (0..n) |j| {
            for (0..j + 1) |i| {
                const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 17 + j * 13 + 1) % 1024)))) * 0.001;
                a.data[j * a.stride + i] = v;
            }
        }

        const y_blazt = try std.testing.allocator.dupe(f32, y0);
        defer std.testing.allocator.free(y_blazt);
        const y_oracle = try std.testing.allocator.dupe(f32, y0);
        defer std.testing.allocator.free(y_oracle);

        blazt.ops.symv(f32, .col_major, .upper, n, alpha, a, x, beta, y_blazt);

        const uplo: u8 = 'U';
        var lda: c_int = @intCast(a.stride);
        ssymv(&uplo, &n_i, &alpha_v, a.data.ptr, &lda, x.ptr, &one, &beta_v, y_oracle.ptr, &one);

        try fx.expectSliceApproxEq(f32, y_oracle, y_blazt, tol);
    }

    // lower
    {
        var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memset(a.data, nan);
        for (0..n) |j| {
            for (j..n) |i| {
                const v = @as(f32, @floatFromInt(@as(u32, @intCast((i * 11 + j * 19 + 3) % 1024)))) * 0.001;
                a.data[j * a.stride + i] = v;
            }
        }

        const y_blazt = try std.testing.allocator.dupe(f32, y0);
        defer std.testing.allocator.free(y_blazt);
        const y_oracle = try std.testing.allocator.dupe(f32, y0);
        defer std.testing.allocator.free(y_oracle);

        blazt.ops.symv(f32, .col_major, .lower, n, alpha, a, x, beta, y_blazt);

        const uplo: u8 = 'L';
        var lda: c_int = @intCast(a.stride);
        ssymv(&uplo, &n_i, &alpha_v, a.data.ptr, &lda, x.ptr, &one, &beta_v, y_oracle.ptr, &one);

        try fx.expectSliceApproxEq(f32, y_oracle, y_blazt, tol);
    }
}

test "BLAS2 parity: hemv (col_major, upper/lower) for Complex(f32)" {
    var oracle = blazt.oracle.Oracle.loadAny(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    const chemv = lookupFn(&oracle.lib, F77ChemvFn, "chemv_") orelse return error.SkipZigTest;
    const tol = fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 };

    const one: c_int = 1;
    const n: usize = 7;
    const n_i: c_int = @intCast(n);

    const alpha: ComplexF32 = ComplexF32.init(0.5, 0.25);
    const beta: ComplexF32 = ComplexF32.init(0.25, -0.125);
    var alpha_v = alpha;
    var beta_v = beta;

    const x = try std.testing.allocator.alloc(ComplexF32, n);
    defer std.testing.allocator.free(x);
    const y0 = try std.testing.allocator.alloc(ComplexF32, n);
    defer std.testing.allocator.free(y0);

    for (x, 0..) |*v, i| {
        const re = @as(f32, @floatFromInt(@as(u32, @intCast((i + 5) % 1024)))) * 0.001;
        const im = @as(f32, @floatFromInt(@as(u32, @intCast((i * 3) % 1024)))) * 0.001;
        v.* = ComplexF32.init(re, im);
    }
    for (y0, 0..) |*v, i| {
        const re = @as(f32, @floatFromInt(@as(u32, @intCast((i + 11) % 1024)))) * 0.002;
        const im = @as(f32, @floatFromInt(@as(u32, @intCast((i * 7) % 1024)))) * 0.002;
        v.* = ComplexF32.init(re, im);
    }

    const nan = std.math.nan(f32);
    const nan_c: ComplexF32 = ComplexF32.init(nan, nan);

    // upper
    {
        var a = try blazt.Matrix(ComplexF32, .col_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memset(a.data, nan_c);

        for (0..n) |j| {
            for (0..j + 1) |i| {
                if (i == j) {
                    // diagonal imag ignored
                    a.data[j * a.stride + i] = ComplexF32.init(1.0, nan);
                } else {
                    const re = @as(f32, @floatFromInt(@as(u32, @intCast((i * 13 + j * 17 + 1) % 1024)))) * 0.001;
                    const im = @as(f32, @floatFromInt(@as(u32, @intCast((i * 19 + j * 23 + 3) % 1024)))) * 0.001;
                    a.data[j * a.stride + i] = ComplexF32.init(re, im);
                }
            }
        }

        const y_blazt = try std.testing.allocator.dupe(ComplexF32, y0);
        defer std.testing.allocator.free(y_blazt);
        const y_oracle = try std.testing.allocator.dupe(ComplexF32, y0);
        defer std.testing.allocator.free(y_oracle);

        blazt.ops.hemv(ComplexF32, .col_major, .upper, n, alpha, a, x, beta, y_blazt);

        const uplo: u8 = 'U';
        var lda: c_int = @intCast(a.stride);
        chemv(&uplo, &n_i, &alpha_v, a.data.ptr, &lda, x.ptr, &one, &beta_v, y_oracle.ptr, &one);

        try expectComplexSliceApproxEq(y_oracle, y_blazt, tol);
    }

    // lower
    {
        var a = try blazt.Matrix(ComplexF32, .col_major).init(std.testing.allocator, n, n);
        defer a.deinit();
        @memset(a.data, nan_c);

        for (0..n) |j| {
            for (j..n) |i| {
                if (i == j) {
                    a.data[j * a.stride + i] = ComplexF32.init(1.0, nan);
                } else {
                    const re = @as(f32, @floatFromInt(@as(u32, @intCast((i * 11 + j * 29 + 5) % 1024)))) * 0.001;
                    const im = @as(f32, @floatFromInt(@as(u32, @intCast((i * 7 + j * 31 + 7) % 1024)))) * 0.001;
                    a.data[j * a.stride + i] = ComplexF32.init(re, -im); // store lower
                }
            }
        }

        const y_blazt = try std.testing.allocator.dupe(ComplexF32, y0);
        defer std.testing.allocator.free(y_blazt);
        const y_oracle = try std.testing.allocator.dupe(ComplexF32, y0);
        defer std.testing.allocator.free(y_oracle);

        blazt.ops.hemv(ComplexF32, .col_major, .lower, n, alpha, a, x, beta, y_blazt);

        const uplo: u8 = 'L';
        var lda: c_int = @intCast(a.stride);
        chemv(&uplo, &n_i, &alpha_v, a.data.ptr, &lda, x.ptr, &one, &beta_v, y_oracle.ptr, &one);

        try expectComplexSliceApproxEq(y_oracle, y_blazt, tol);
    }
}


