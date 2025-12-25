const std = @import("std");
const blazt = @import("blazt");

fn fillZero(slice: []f32) void {
    @memset(slice, 0);
}

test "ops.ger row_major updates A += alpha * x * y^T" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillZero(a.data);

    const x = [_]f32{ 1, 2 };
    const y = [_]f32{ 3, 4, 5 };

    blazt.ops.ger(f32, .row_major, 2, 3, 2.0, &x, &y, &a);

    try std.testing.expectEqualSlices(f32, &[_]f32{
        6, 8, 10,
        12, 16, 20,
    }, a.data);
}

test "ops.ger col_major matches row_major result" {
    var a = try blazt.Matrix(f32, .col_major).init(std.testing.allocator, 2, 3);
    defer a.deinit();
    fillZero(a.data);

    const x = [_]f32{ 1, 2 };
    const y = [_]f32{ 3, 4, 5 };

    blazt.ops.ger(f32, .col_major, 2, 3, 2.0, &x, &y, &a);

    // A = [[6,8,10],[12,16,20]] in col-major storage.
    try std.testing.expectEqualSlices(f32, &[_]f32{
        6, 12,
        8, 16,
        10, 20,
    }, a.data);
}

test "ops.ger accumulates into existing A" {
    var a = try blazt.Matrix(f32, .row_major).init(std.testing.allocator, 2, 2);
    defer a.deinit();
    @memset(a.data, 1);

    const x = [_]f32{ 1, 1 };
    const y = [_]f32{ 1, 1 };

    blazt.ops.ger(f32, .row_major, 2, 2, 2.0, &x, &y, &a);
    // A was all ones; add 2*(1*1) = 2 everywhere => 3
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 3, 3, 3 }, a.data);
}


