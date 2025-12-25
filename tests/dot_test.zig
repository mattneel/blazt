const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn dotRefF64(n: usize, x: []const f32, y: []const f32) f32 {
    var acc: f64 = 0.0;
    for (0..n) |i| {
        acc += @as(f64, @floatCast(x[i])) * @as(f64, @floatCast(y[i]));
    }
    return @floatCast(acc);
}

test "ops.dot matches simple known case" {
    const x: [3]f32 = .{ 1.0, 2.0, 3.0 };
    const y: [3]f32 = .{ 4.0, 5.0, 6.0 };
    const got = blazt.ops.dot(f32, &x, &y);
    try std.testing.expectEqual(@as(f32, 32.0), got);
}

test "ops.dot matches reference within tolerance (odd size + tail)" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const n: usize = vl * 3 + 1;

    var rng = fx.FixtureRng.init(0xD070_0001);
    const r = rng.random();

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);

    fx.fillSliceRandom(r, x);
    fx.fillSliceRandom(r, y);

    const ref = dotRefF64(n, x, y);
    const got = blazt.ops.dot(f32, x, y);
    try fx.expectFloatApproxEq(f32, ref, got, fx.defaultTolerance(f32));
}

test "ops.dot large-n stress matches reference within tolerance" {
    const n: usize = 1 << 20;

    var rng = fx.FixtureRng.init(0xD070_0002);
    const r = rng.random();

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);

    fx.fillSliceRandom(r, x);
    fx.fillSliceRandom(r, y);

    const ref = dotRefF64(n, x, y);
    const got = blazt.ops.dot(f32, x, y);
    try fx.expectFloatApproxEq(f32, ref, got, fx.defaultTolerance(f32));
}


