const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn fillMatDeterministic(comptime T: type, comptime layout: blazt.Layout, mat: *blazt.Matrix(T, layout)) void {
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const s: u32 = @intCast((i * 131 + j * 17 + 1) % 1024);
            mat.atPtr(i, j).* = @as(T, @floatFromInt(s)) * @as(T, 0.001);
        }
    }
}

fn fillMatDeterministicComplex(
    comptime C: type,
    comptime layout: blazt.Layout,
    mat: *blazt.Matrix(C, layout),
) void {
    const R = @TypeOf(@as(C, undefined).re);
    for (0..mat.rows) |i| {
        for (0..mat.cols) |j| {
            const a: R = @as(R, @floatFromInt(@as(u32, @intCast((i * 19 + j * 23 + 7) % 257)))) * @as(R, 0.01);
            const b: R = @as(R, @floatFromInt(@as(u32, @intCast((i * 11 + j * 29 + 3) % 257)))) * @as(R, 0.01);
            mat.atPtr(i, j).* = C.init(a, b);
        }
    }
}

fn copyMat(comptime T: type, comptime layout: blazt.Layout, dst: *blazt.Matrix(T, layout), src: blazt.Matrix(T, layout)) void {
    std.debug.assert(dst.rows == src.rows and dst.cols == src.cols and dst.stride == src.stride);
    @memcpy(dst.data, src.data);
}

fn refSyrk(
    comptime T: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: blazt.Matrix(T, layout),
    beta: T,
    c0: blazt.Matrix(T, layout),
    c_out: *blazt.Matrix(T, layout),
) void {
    copyMat(T, layout, c_out, c0);

    const eff_trans: blazt.Trans = switch (trans) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };

    const aAt = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    const aOp = struct {
        inline fn f(mat: blazt.Matrix(T, layout), tr: blazt.Trans, i: usize, p: usize) T {
            return if (tr == .no_trans) aAt(mat, i, p) else aAt(mat, p, i);
        }
    }.f;

    switch (uplo) {
        .upper => {
            for (0..n) |j| {
                for (0..j + 1) |i| {
                    var sum: T = 0;
                    for (0..k) |p| {
                        sum += aOp(a, eff_trans, i, p) * aOp(a, eff_trans, j, p);
                    }
                    const cp = c_out.atPtr(i, j);
                    cp.* = alpha * sum + beta * cp.*;
                }
            }
        },
        .lower => {
            for (0..n) |j| {
                for (j..n) |i| {
                    var sum: T = 0;
                    for (0..k) |p| {
                        sum += aOp(a, eff_trans, i, p) * aOp(a, eff_trans, j, p);
                    }
                    const cp = c_out.atPtr(i, j);
                    cp.* = alpha * sum + beta * cp.*;
                }
            }
        },
    }
}

test "ops.syrk updates only requested triangle (uplo) and matches reference (row/col major, trans)" {
    const T = f32;
    const alpha: T = 0.75;
    const beta: T = 0.25;
    const tol = fx.FloatTolerance{ .abs = 1e-5, .rel = 1e-5, .ulps = 256 };

    const n: usize = 19;
    const k: usize = 13;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
            inline for (.{ blazt.Trans.no_trans, blazt.Trans.trans, blazt.Trans.conj_trans }) |tr| {
                const eff: blazt.Trans = switch (tr) {
                    .no_trans => .no_trans,
                    .trans, .conj_trans => .trans,
                };

                const a_rows: usize = if (eff == .no_trans) n else k;
                const a_cols: usize = if (eff == .no_trans) k else n;

                var a = try blazt.Matrix(T, layout).init(std.testing.allocator, a_rows, a_cols);
                defer a.deinit();
                var c = try blazt.Matrix(T, layout).init(std.testing.allocator, n, n);
                defer c.deinit();
                var c0 = try blazt.Matrix(T, layout).init(std.testing.allocator, n, n);
                defer c0.deinit();
                var cref = try blazt.Matrix(T, layout).init(std.testing.allocator, n, n);
                defer cref.deinit();

                fillMatDeterministic(T, layout, &a);
                fillMatDeterministic(T, layout, &c);
                copyMat(T, layout, &c0, c);

                blazt.ops.syrk(T, layout, uplo, tr, n, k, alpha, a, beta, &c);
                refSyrk(T, layout, uplo, tr, n, k, alpha, a, beta, c0, &cref);

                // Triangle matches ref; other triangle unchanged vs original.
                for (0..n) |j| {
                    for (0..n) |i| {
                        const in_tri = switch (uplo) {
                            .upper => i <= j,
                            .lower => i >= j,
                        };
                        if (in_tri) {
                            try fx.expectFloatApproxEq(T, cref.at(i, j), c.at(i, j), tol);
                        } else {
                            try std.testing.expectEqual(c0.at(i, j), c.at(i, j));
                        }
                    }
                }
            }
        }
    }
}

fn refSyr2k(
    comptime T: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: blazt.Matrix(T, layout),
    b: blazt.Matrix(T, layout),
    beta: T,
    c0: blazt.Matrix(T, layout),
    c_out: *blazt.Matrix(T, layout),
) void {
    copyMat(T, layout, c_out, c0);

    const eff_trans: blazt.Trans = switch (trans) {
        .no_trans => .no_trans,
        .trans, .conj_trans => .trans,
    };

    const at = struct {
        inline fn f(mat: blazt.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    const op = struct {
        inline fn f(mat: blazt.Matrix(T, layout), tr: blazt.Trans, i: usize, p: usize) T {
            return if (tr == .no_trans) at(mat, i, p) else at(mat, p, i);
        }
    }.f;

    switch (uplo) {
        .upper => {
            for (0..n) |j| {
                for (0..j + 1) |i| {
                    var sum: T = 0;
                    for (0..k) |p| {
                        sum += op(a, eff_trans, i, p) * op(b, eff_trans, j, p);
                        sum += op(b, eff_trans, i, p) * op(a, eff_trans, j, p);
                    }
                    const cp = c_out.atPtr(i, j);
                    cp.* = alpha * sum + beta * cp.*;
                }
            }
        },
        .lower => {
            for (0..n) |j| {
                for (j..n) |i| {
                    var sum: T = 0;
                    for (0..k) |p| {
                        sum += op(a, eff_trans, i, p) * op(b, eff_trans, j, p);
                        sum += op(b, eff_trans, i, p) * op(a, eff_trans, j, p);
                    }
                    const cp = c_out.atPtr(i, j);
                    cp.* = alpha * sum + beta * cp.*;
                }
            }
        },
    }
}

test "ops.syr2k updates only requested triangle (uplo) and matches reference (row/col major, trans)" {
    const T = f32;
    const alpha: T = 0.5;
    const beta: T = 0.25;
    const tol = fx.FloatTolerance{ .abs = 1e-5, .rel = 1e-5, .ulps = 256 };

    const n: usize = 17;
    const k: usize = 9;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
            inline for (.{ blazt.Trans.no_trans, blazt.Trans.trans, blazt.Trans.conj_trans }) |tr| {
                const eff: blazt.Trans = switch (tr) {
                    .no_trans => .no_trans,
                    .trans, .conj_trans => .trans,
                };

                const a_rows: usize = if (eff == .no_trans) n else k;
                const a_cols: usize = if (eff == .no_trans) k else n;

                var a = try blazt.Matrix(T, layout).init(std.testing.allocator, a_rows, a_cols);
                defer a.deinit();
                var b = try blazt.Matrix(T, layout).init(std.testing.allocator, a_rows, a_cols);
                defer b.deinit();
                var c = try blazt.Matrix(T, layout).init(std.testing.allocator, n, n);
                defer c.deinit();
                var c0 = try blazt.Matrix(T, layout).init(std.testing.allocator, n, n);
                defer c0.deinit();
                var cref = try blazt.Matrix(T, layout).init(std.testing.allocator, n, n);
                defer cref.deinit();

                fillMatDeterministic(T, layout, &a);
                // Offset B slightly so it's not identical.
                fillMatDeterministic(T, layout, &b);
                for (b.data, 0..) |*v, idx| v.* += @as(T, @floatFromInt(@as(u32, @intCast(idx % 7)))) * @as(T, 0.0001);

                fillMatDeterministic(T, layout, &c);
                copyMat(T, layout, &c0, c);

                blazt.ops.syr2k(T, layout, uplo, tr, n, k, alpha, a, b, beta, &c);
                refSyr2k(T, layout, uplo, tr, n, k, alpha, a, b, beta, c0, &cref);

                for (0..n) |j| {
                    for (0..n) |i| {
                        const in_tri = switch (uplo) {
                            .upper => i <= j,
                            .lower => i >= j,
                        };
                        if (in_tri) {
                            try fx.expectFloatApproxEq(T, cref.at(i, j), c.at(i, j), tol);
                        } else {
                            try std.testing.expectEqual(c0.at(i, j), c.at(i, j));
                        }
                    }
                }
            }
        }
    }
}

fn refHerk(
    comptime C: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    n: usize,
    k: usize,
    alpha: @TypeOf(@as(C, undefined).re),
    a: blazt.Matrix(C, layout),
    beta: @TypeOf(@as(C, undefined).re),
    c0: blazt.Matrix(C, layout),
    c_out: *blazt.Matrix(C, layout),
) void {
    copyMat(C, layout, c_out, c0);

    const R = @TypeOf(@as(C, undefined).re);
    const alpha_c: C = C.init(alpha, @as(R, 0));
    const beta_c: C = C.init(beta, @as(R, 0));
    const beta_is_zero = beta == @as(R, 0);

    const eff_trans: blazt.Trans = switch (trans) {
        .no_trans => .no_trans,
        .conj_trans, .trans => .conj_trans,
    };

    const aAt = struct {
        inline fn f(mat: blazt.Matrix(C, layout), i: usize, j: usize) C {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    switch (uplo) {
        .upper => for (0..n) |j| for (0..j + 1) |i| {
            var sum: C = C.init(@as(R, 0), @as(R, 0));
            if (eff_trans == .no_trans) {
                for (0..k) |p| sum = sum.add(aAt(a, i, p).mul(aAt(a, j, p).conjugate()));
            } else {
                for (0..k) |p| sum = sum.add(aAt(a, p, i).conjugate().mul(aAt(a, p, j)));
            }
            const scaled = alpha_c.mul(sum);
            const cp = c_out.atPtr(i, j);
            const out = if (beta_is_zero) scaled else scaled.add(cp.*.mul(beta_c));
            cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
        },
        .lower => for (0..n) |j| for (j..n) |i| {
            var sum: C = C.init(@as(R, 0), @as(R, 0));
            if (eff_trans == .no_trans) {
                for (0..k) |p| sum = sum.add(aAt(a, i, p).mul(aAt(a, j, p).conjugate()));
            } else {
                for (0..k) |p| sum = sum.add(aAt(a, p, i).conjugate().mul(aAt(a, p, j)));
            }
            const scaled = alpha_c.mul(sum);
            const cp = c_out.atPtr(i, j);
            const out = if (beta_is_zero) scaled else scaled.add(cp.*.mul(beta_c));
            cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
        },
    }
}

test "ops.herk updates only requested triangle (uplo), enforces real diagonal, and matches reference" {
    const C = std.math.Complex(f32);
    const R = f32;
    const alpha: R = 0.5;
    const beta: R = 0.25;
    const tol = fx.FloatTolerance{ .abs = 1e-4, .rel = 1e-4, .ulps = 512 };

    const n: usize = 11;
    const k: usize = 7;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
            inline for (.{ blazt.Trans.no_trans, blazt.Trans.conj_trans, blazt.Trans.trans }) |tr| {
                const eff: blazt.Trans = switch (tr) {
                    .no_trans => .no_trans,
                    .conj_trans, .trans => .conj_trans,
                };
                const a_rows: usize = if (eff == .no_trans) n else k;
                const a_cols: usize = if (eff == .no_trans) k else n;

                var a = try blazt.Matrix(C, layout).init(std.testing.allocator, a_rows, a_cols);
                defer a.deinit();
                var c = try blazt.Matrix(C, layout).init(std.testing.allocator, n, n);
                defer c.deinit();
                var c0 = try blazt.Matrix(C, layout).init(std.testing.allocator, n, n);
                defer c0.deinit();
                var cref = try blazt.Matrix(C, layout).init(std.testing.allocator, n, n);
                defer cref.deinit();

                fillMatDeterministicComplex(C, layout, &a);
                fillMatDeterministicComplex(C, layout, &c);
                // Poison diagonal imag; herk must zero it for updated diagonal entries.
                for (0..n) |i| c.atPtr(i, i).* = C.init(c.at(i, i).re, 123.0);

                copyMat(C, layout, &c0, c);

                blazt.ops.herk(C, layout, uplo, tr, n, k, alpha, a, beta, &c);
                refHerk(C, layout, uplo, tr, n, k, alpha, a, beta, c0, &cref);

                for (0..n) |j| {
                    for (0..n) |i| {
                        const in_tri = switch (uplo) {
                            .upper => i <= j,
                            .lower => i >= j,
                        };
                        const got = c.at(i, j);
                        if (in_tri) {
                            const exp = cref.at(i, j);
                            try fx.expectFloatApproxEq(R, exp.re, got.re, tol);
                            try fx.expectFloatApproxEq(R, exp.im, got.im, tol);
                            if (i == j) try std.testing.expectEqual(@as(R, 0), got.im);
                        } else {
                            // untouched triangle
                            try std.testing.expectEqual(c0.at(i, j), got);
                        }
                    }
                }
            }
        }
    }
}

fn refHer2k(
    comptime C: type,
    comptime layout: blazt.Layout,
    uplo: blazt.UpLo,
    trans: blazt.Trans,
    n: usize,
    k: usize,
    alpha: C,
    a: blazt.Matrix(C, layout),
    b: blazt.Matrix(C, layout),
    beta: @TypeOf(@as(C, undefined).re),
    c0: blazt.Matrix(C, layout),
    c_out: *blazt.Matrix(C, layout),
) void {
    copyMat(C, layout, c_out, c0);

    const R = @TypeOf(@as(C, undefined).re);
    const zero: C = C.init(@as(R, 0), @as(R, 0));
    const beta_c: C = C.init(beta, @as(R, 0));
    const alpha_conj: C = alpha.conjugate();
    const beta_is_zero = beta == @as(R, 0);

    const eff_trans: blazt.Trans = switch (trans) {
        .no_trans => .no_trans,
        .conj_trans, .trans => .conj_trans,
    };

    const aAt = struct {
        inline fn f(mat: blazt.Matrix(C, layout), i: usize, j: usize) C {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    switch (uplo) {
        .upper => for (0..n) |j| for (0..j + 1) |i| {
            var sum1: C = zero;
            var sum2: C = zero;
            if (eff_trans == .no_trans) {
                for (0..k) |p| {
                    sum1 = sum1.add(aAt(a, i, p).mul(aAt(b, j, p).conjugate()));
                    sum2 = sum2.add(aAt(b, i, p).mul(aAt(a, j, p).conjugate()));
                }
            } else {
                for (0..k) |p| {
                    sum1 = sum1.add(aAt(a, p, i).conjugate().mul(aAt(b, p, j)));
                    sum2 = sum2.add(aAt(b, p, i).conjugate().mul(aAt(a, p, j)));
                }
            }
            var out: C = alpha.mul(sum1).add(alpha_conj.mul(sum2));
            const cp = c_out.atPtr(i, j);
            if (!beta_is_zero) out = out.add(cp.*.mul(beta_c));
            cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
        },
        .lower => for (0..n) |j| for (j..n) |i| {
            var sum1: C = zero;
            var sum2: C = zero;
            if (eff_trans == .no_trans) {
                for (0..k) |p| {
                    sum1 = sum1.add(aAt(a, i, p).mul(aAt(b, j, p).conjugate()));
                    sum2 = sum2.add(aAt(b, i, p).mul(aAt(a, j, p).conjugate()));
                }
            } else {
                for (0..k) |p| {
                    sum1 = sum1.add(aAt(a, p, i).conjugate().mul(aAt(b, p, j)));
                    sum2 = sum2.add(aAt(b, p, i).conjugate().mul(aAt(a, p, j)));
                }
            }
            var out: C = alpha.mul(sum1).add(alpha_conj.mul(sum2));
            const cp = c_out.atPtr(i, j);
            if (!beta_is_zero) out = out.add(cp.*.mul(beta_c));
            cp.* = if (i == j) C.init(out.re, @as(R, 0)) else out;
        },
    }
}

test "ops.her2k updates only requested triangle (uplo), enforces real diagonal, and matches reference" {
    const C = std.math.Complex(f32);
    const R = f32;
    const alpha: C = C.init(0.25, -0.5);
    const beta: R = 0.125;
    const tol = fx.FloatTolerance{ .abs = 2e-4, .rel = 2e-4, .ulps = 1024 };

    const n: usize = 9;
    const k: usize = 6;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        inline for (.{ blazt.UpLo.upper, blazt.UpLo.lower }) |uplo| {
            inline for (.{ blazt.Trans.no_trans, blazt.Trans.conj_trans, blazt.Trans.trans }) |tr| {
                const eff: blazt.Trans = switch (tr) {
                    .no_trans => .no_trans,
                    .conj_trans, .trans => .conj_trans,
                };
                const a_rows: usize = if (eff == .no_trans) n else k;
                const a_cols: usize = if (eff == .no_trans) k else n;

                var a = try blazt.Matrix(C, layout).init(std.testing.allocator, a_rows, a_cols);
                defer a.deinit();
                var b = try blazt.Matrix(C, layout).init(std.testing.allocator, a_rows, a_cols);
                defer b.deinit();
                var c = try blazt.Matrix(C, layout).init(std.testing.allocator, n, n);
                defer c.deinit();
                var c0 = try blazt.Matrix(C, layout).init(std.testing.allocator, n, n);
                defer c0.deinit();
                var cref = try blazt.Matrix(C, layout).init(std.testing.allocator, n, n);
                defer cref.deinit();

                fillMatDeterministicComplex(C, layout, &a);
                fillMatDeterministicComplex(C, layout, &b);
                // Offset B a bit.
                const RR = @TypeOf(@as(C, undefined).re);
                for (b.data, 0..) |*v, idx| v.* = C.init(v.*.re + @as(RR, @floatFromInt(@as(u32, @intCast(idx % 5)))) * @as(RR, 0.001), v.*.im);

                fillMatDeterministicComplex(C, layout, &c);
                for (0..n) |i| c.atPtr(i, i).* = C.init(c.at(i, i).re, 999.0);
                copyMat(C, layout, &c0, c);

                blazt.ops.her2k(C, layout, uplo, tr, n, k, alpha, a, b, beta, &c);
                refHer2k(C, layout, uplo, tr, n, k, alpha, a, b, beta, c0, &cref);

                for (0..n) |j| {
                    for (0..n) |i| {
                        const in_tri = switch (uplo) {
                            .upper => i <= j,
                            .lower => i >= j,
                        };
                        const got = c.at(i, j);
                        if (in_tri) {
                            const exp = cref.at(i, j);
                            try fx.expectFloatApproxEq(R, exp.re, got.re, tol);
                            try fx.expectFloatApproxEq(R, exp.im, got.im, tol);
                            if (i == j) try std.testing.expectEqual(@as(R, 0), got.im);
                        } else {
                            try std.testing.expectEqual(c0.at(i, j), got);
                        }
                    }
                }
            }
        }
    }
}


