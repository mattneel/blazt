const std = @import("std");
const blazt = @import("blazt");

fn fillUpperRowMajor(a: *blazt.Matrix(f32, .row_major)) void {
    // A = [[1,2,3],[0,4,5],[0,0,6]]
    a.data[0] = 1; a.data[1] = 2; a.data[2] = 3;
    a.data[3] = 0; a.data[4] = 4; a.data[5] = 5;
    a.data[6] = 0; a.data[7] = 0; a.data[8] = 6;
}

fn fillUpperColMajor(a: *blazt.Matrix(f32, .col_major)) void {
    // Column-major storage of same upper-triangular matrix.
    // col0: [1,0,0], col1: [2,4,0], col2: [3,5,6]
    a.data[0] = 1; a.data[1] = 0; a.data[2] = 0;
    a.data[3] = 2; a.data[4] = 4; a.data[5] = 0;
    a.data[6] = 3; a.data[7] = 5; a.data[8] = 6;
}

test "ops.trmv upper no_trans non_unit (row_major)" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();
    fillUpperRowMajor(&a);

    var x = [_]f32{ 1, 1, 1 };
    blazt.ops.trmv(f32, .row_major, .upper, .no_trans, .non_unit, 3, a, &x);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 6, 9, 6 }, &x);
}

test "ops.trmv upper trans non_unit (col_major)" {
    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();
    fillUpperColMajor(&a);

    var x = [_]f32{ 1, 1, 1 };
    blazt.ops.trmv(f32, .col_major, .upper, .trans, .non_unit, 3, a, &x);
    // A^T * [1,1,1] = [1, 6, 14]
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 6, 14 }, &x);
}

test "ops.trmv unit diagonal does not read diagonal (NaN-safe)" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();
    fillUpperRowMajor(&a);

    // Poison the diagonal with NaNs. If trmv reads them, the result will become NaN.
    a.data[0] = std.math.nan(f32);
    a.data[4] = std.math.nan(f32);
    a.data[8] = std.math.nan(f32);

    var x = [_]f32{ 1, 1, 1 };
    blazt.ops.trmv(f32, .row_major, .upper, .no_trans, .unit, 3, a, &x);
    // With unit diag: y0 = 1 + 2 + 3 = 6, y1 = 1 + 5 = 6, y2 = 1
    try std.testing.expectEqualSlices(f32, &[_]f32{ 6, 6, 1 }, &x);
    for (x) |v| try std.testing.expect(!std.math.isNan(v));
}

test "ops.trsv upper no_trans non_unit solves correctly" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();
    fillUpperRowMajor(&a);

    // x_true = [1,2,3], b = A*x_true = [14,23,18]
    var x = [_]f32{ 14, 23, 18 };
    try blazt.ops.trsv(f32, .row_major, .upper, .no_trans, .non_unit, 3, a, &x);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3 }, &x);
}

test "ops.trsv upper trans non_unit solves correctly" {
    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, 3, 3);
    defer a.deinit();
    fillUpperColMajor(&a);

    // Solve A^T x = b for x_true=[1,2,3]; b = [1,10,31]
    var x = [_]f32{ 1, 10, 31 };
    try blazt.ops.trsv(f32, .col_major, .upper, .trans, .non_unit, 3, a, &x);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3 }, &x);
}

test "ops.trsv singular (zero diagonal) returns error.Singular for non_unit" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 2);
    defer a.deinit();
    // upper with zero diagonal
    a.data[0] = 0; a.data[1] = 1;
    a.data[2] = 0; a.data[3] = 2;

    var x = [_]f32{ 1, 2 };
    try std.testing.expectError(error.Singular, blazt.ops.trsv(f32, .row_major, .upper, .no_trans, .non_unit, 2, a, &x));
}

test "ops.trsv unit diagonal ignores zero diagonal (does not error)" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 2);
    defer a.deinit();
    // lower with zero diagonal but unit diag should treat as 1.
    a.data[0] = 0; a.data[1] = 0;
    a.data[2] = 3; a.data[3] = 0;

    // Solve A*x=b with unit diag:
    // A = [[1,0],[3,1]] (effective), b=[1,2] -> x0=1, x1=2-3*1=-1
    var x = [_]f32{ 1, 2 };
    try blazt.ops.trsv(f32, .row_major, .lower, .no_trans, .unit, 2, a, &x);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, -1 }, &x);
}


