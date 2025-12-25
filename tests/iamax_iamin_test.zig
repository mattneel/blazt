const std = @import("std");
const blazt = @import("blazt");

test "ops.iamax returns first max-abs on ties" {
    const x: [4]f32 = .{ -2.0, 2.0, -2.0, 1.0 };
    const got = blazt.ops.iamax(f32, &x) orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(@as(usize, 0), got);
}

test "ops.iamin returns first min-abs on ties" {
    const x: [4]f32 = .{ -1.0, 1.0, -1.0, 2.0 };
    const got = blazt.ops.iamin(f32, &x) orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(@as(usize, 0), got);
}

test "ops.iamax/iamin basic correctness" {
    const x: [5]f32 = .{ 3.0, -4.0, 2.0, -0.5, 1.0 };
    try std.testing.expectEqual(@as(usize, 1), blazt.ops.iamax(f32, &x).?); // | -4 | = 4
    try std.testing.expectEqual(@as(usize, 3), blazt.ops.iamin(f32, &x).?); // | -0.5 | = 0.5
}

test "ops.iamax/iamin NaN policy: first NaN wins" {
    const nan = std.math.nan(f32);
    const x: [4]f32 = .{ 1.0, nan, 10.0, nan };
    try std.testing.expectEqual(@as(usize, 1), blazt.ops.iamax(f32, &x).?);
    try std.testing.expectEqual(@as(usize, 1), blazt.ops.iamin(f32, &x).?);
}

test "ops.iamax/iamin empty slice returns null" {
    const x: [0]f32 = .{};
    try std.testing.expect(blazt.ops.iamax(f32, &x) == null);
    try std.testing.expect(blazt.ops.iamin(f32, &x) == null);
}


