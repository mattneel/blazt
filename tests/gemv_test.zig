const std = @import("std");
const blazt = @import("blazt");

fn fillA23RowMajor(a: *blazt.Matrix(f32, .row_major)) void {
    // A = [[1,2,3],[4,5,6]] in row-major order
    a.data[0] = 1; a.data[1] = 2; a.data[2] = 3;
    a.data[3] = 4; a.data[4] = 5; a.data[5] = 6;
}

fn fillA23ColMajor(a: *blazt.Matrix(f32, .col_major)) void {
    // A = [[1,2,3],[4,5,6]] in column-major order (cols packed)
    a.data[0] = 1; a.data[1] = 4; // col 0
    a.data[2] = 2; a.data[3] = 5; // col 1
    a.data[4] = 3; a.data[5] = 6; // col 2
}

test "ops.gemv row_major no_trans beta=0 does not read y" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillA23RowMajor(&a);

    const x = [_]f32{ 10, 20, 30 };
    var y = [_]f32{ std.math.nan(f32), std.math.nan(f32) };

    blazt.ops.gemv(f32, .row_major, .no_trans, 2, 3, 1.0, a, &x, 0.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 140, 320 }, &y);
}

test "ops.gemv row_major trans beta=0 does not read y" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillA23RowMajor(&a);

    const x = [_]f32{ 10, 20 };
    var y = [_]f32{ std.math.nan(f32), std.math.nan(f32), std.math.nan(f32) };

    blazt.ops.gemv(f32, .row_major, .trans, 2, 3, 1.0, a, &x, 0.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 90, 120, 150 }, &y);
}

test "ops.gemv col_major no_trans matches row_major result" {
    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillA23ColMajor(&a);

    const x = [_]f32{ 10, 20, 30 };
    var y = [_]f32{ std.math.nan(f32), std.math.nan(f32) };

    blazt.ops.gemv(f32, .col_major, .no_trans, 2, 3, 1.0, a, &x, 0.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 140, 320 }, &y);
}

test "ops.gemv col_major trans matches row_major trans result" {
    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillA23ColMajor(&a);

    const x = [_]f32{ 10, 20 };
    var y = [_]f32{ std.math.nan(f32), std.math.nan(f32), std.math.nan(f32) };

    blazt.ops.gemv(f32, .col_major, .conj_trans, 2, 3, 1.0, a, &x, 0.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 90, 120, 150 }, &y);
}

test "ops.gemv beta=1 accumulates into y" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillA23RowMajor(&a);

    const x = [_]f32{ 10, 20, 30 };
    var y = [_]f32{ 1, 2 };

    blazt.ops.gemv(f32, .row_major, .no_trans, 2, 3, 1.0, a, &x, 1.0, &y);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 141, 322 }, &y);
}


