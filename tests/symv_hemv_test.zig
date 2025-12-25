const std = @import("std");
const blazt = @import("blazt");

test "ops.symv upper uses only upper triangle (NaN-safe)" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();

    const nan = std.math.nan(f32);
    // Fill everything with NaN then overwrite the upper triangle.
    @memset(a.data, nan);
    // A = [[1,2,3],[2,4,5],[3,5,6]]
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;
    a.data[4] = 4;
    a.data[5] = 5;
    a.data[8] = 6;
    // Upper off-diagonals
    a.data[3] = nan; // (1,0) unused lower
    a.data[6] = nan; // (2,0) unused lower
    a.data[7] = nan; // (2,1) unused lower

    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ nan, nan, nan };

    blazt.ops.symv(f32, .row_major, .upper, 3, 1.0, a, &x, 0.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 14, 25, 31 }, &y);
    for (y) |v| try std.testing.expect(!std.math.isNan(v));
}

test "ops.symv lower uses only lower triangle (NaN-safe) (col_major)" {
    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();

    const nan = std.math.nan(f32);
    @memset(a.data, nan);

    // Store lower triangle in column-major form.
    // A = [[1,2,3],[2,4,5],[3,5,6]]; lower contains (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)
    // col0: [1,2,3]
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;
    // col1: [NaN,4,5] (row0 is upper/unused)
    a.data[3] = nan;
    a.data[4] = 4;
    a.data[5] = 5;
    // col2: [NaN,NaN,6]
    a.data[6] = nan;
    a.data[7] = nan;
    a.data[8] = 6;

    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ nan, nan, nan };

    blazt.ops.symv(f32, .col_major, .lower, 3, 1.0, a, &x, 0.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 14, 25, 31 }, &y);
    for (y) |v| try std.testing.expect(!std.math.isNan(v));
}

test "ops.hemv upper conjugates off-diagonal and ignores diagonal imag (NaN-safe)" {
    const C = std.math.Complex(f32);
    const nan = std.math.nan(f32);

    var a = try blazt.Matrix(C, .row_major).init(std.testing.allocator, 2, 2);
    defer a.deinit();

    // Fill everything with NaNs then store upper triangle only.
    @memset(a.data, C.init(nan, nan));
    // diag imaginary is ignored, so poison it
    a.data[0] = C.init(1, nan); // (0,0)
    a.data[1] = C.init(2, 3);   // (0,1) stored upper
    a.data[3] = C.init(4, nan); // (1,1)
    // (1,0) is lower/unused -> remains NaN

    const x = [_]C{ C.init(1, 1), C.init(2, -1) };
    var y = [_]C{ C.init(nan, nan), C.init(nan, nan) };

    blazt.ops.hemv(C, .row_major, .upper, 2, C.init(1, 0), a, &x, C.init(0, 0), &y);

    // Expected:
    // y0 = 8 + 5i
    // y1 = 13 - 5i
    try std.testing.expectEqual(@as(f32, 8), y[0].re);
    try std.testing.expectEqual(@as(f32, 5), y[0].im);
    try std.testing.expectEqual(@as(f32, 13), y[1].re);
    try std.testing.expectEqual(@as(f32, -5), y[1].im);

    try std.testing.expect(!std.math.isNan(y[0].re) and !std.math.isNan(y[0].im));
    try std.testing.expect(!std.math.isNan(y[1].re) and !std.math.isNan(y[1].im));
}

test "ops.hemv lower matches upper result (col_major)" {
    const C = std.math.Complex(f32);
    const nan = std.math.nan(f32);

    var a = try blazt.Matrix(C, .col_major).init(std.testing.allocator, 2, 2);
    defer a.deinit();

    @memset(a.data, C.init(nan, nan));
    // lower storage: store (0,0), (1,0), (1,1); poison upper
    a.data[0] = C.init(1, nan);   // col0,row0
    a.data[1] = C.init(2, -3);    // col0,row1 == A(1,0)
    a.data[2] = C.init(nan, nan); // col1,row0 == upper/unused
    a.data[3] = C.init(4, nan);   // col1,row1 diag

    const x = [_]C{ C.init(1, 1), C.init(2, -1) };
    var y = [_]C{ C.init(nan, nan), C.init(nan, nan) };

    blazt.ops.hemv(C, .col_major, .lower, 2, C.init(1, 0), a, &x, C.init(0, 0), &y);

    try std.testing.expectEqual(@as(f32, 8), y[0].re);
    try std.testing.expectEqual(@as(f32, 5), y[0].im);
    try std.testing.expectEqual(@as(f32, 13), y[1].re);
    try std.testing.expectEqual(@as(f32, -5), y[1].im);
}

test "ops.hemv beta=1 accumulates into y" {
    const C = std.math.Complex(f32);

    var a = try blazt.Matrix(C, .row_major).init(std.testing.allocator, 2, 2);
    defer a.deinit();

    // Store full Hermitian for simplicity here.
    a.data[0] = C.init(1, 0);
    a.data[1] = C.init(2, 3);
    a.data[2] = C.init(2, -3);
    a.data[3] = C.init(4, 0);

    const x = [_]C{ C.init(1, 1), C.init(2, -1) };
    var y = [_]C{ C.init(1, 0), C.init(1, 0) };

    blazt.ops.hemv(C, .row_major, .upper, 2, C.init(1, 0), a, &x, C.init(1, 0), &y);

    // Previous expected was [8+5i, 13-5i]; plus initial [1,1] => [9+5i, 14-5i]
    try std.testing.expectEqual(@as(f32, 9), y[0].re);
    try std.testing.expectEqual(@as(f32, 5), y[0].im);
    try std.testing.expectEqual(@as(f32, 14), y[1].re);
    try std.testing.expectEqual(@as(f32, -5), y[1].im);
}


