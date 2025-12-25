const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

test "ops.scal alpha=0 zeros (normal) data" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const n: usize = vl * 3 + 1; // force a scalar tail when vl>1

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i + 1))));

    blazt.ops.scal(f32, n, @as(f32, 0.0), x);

    for (x) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "ops.scal alpha=1 is identity" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const n: usize = vl * 3 + 1;

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(u32, @intCast(i + 1)))) * @as(f32, 0.25);

    const orig = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(orig);

    blazt.ops.scal(f32, n, @as(f32, 1.0), x);
    try fx.expectSliceApproxEq(f32, orig, x, fx.defaultTolerance(f32));
}

test "ops.scal alpha=random scales first n elements and tail handling is correct" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const len: usize = vl * 3 + 1;
    const n: usize = len - 2; // leave an untouched tail in addition to SIMD tail

    var rng = fx.FixtureRng.init(0xdecaf_bad_f00d_beef);

    const x = try std.testing.allocator.alloc(f32, len);
    defer std.testing.allocator.free(x);
    fx.fillSliceRandom(rng.random(), x);

    const expected = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(expected);

    const alpha = fx.randomScalar(rng.random(), f32);
    blazt.ops.scal(f32, n, alpha, x);

    for (0..n) |i| {
        expected[i] = alpha * expected[i];
    }
    // expected[n..] remains unchanged

    try fx.expectSliceApproxEq(f32, expected, x, fx.defaultTolerance(f32));
}


