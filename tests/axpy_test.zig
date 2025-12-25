const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

test "ops.axpy alpha=0 leaves y unchanged" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const n: usize = vl * 3 + 1;

    var rng = fx.FixtureRng.init(0xA8A8_0000);
    const r = rng.random();

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);

    fx.fillSliceRandom(r, x);
    fx.fillSliceRandom(r, y);

    const y_before = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(y_before);

    blazt.ops.axpy(f32, n, @as(f32, 0.0), x, y);
    try std.testing.expectEqualSlices(f32, y_before, y);
}

test "ops.axpy alpha=1 accumulates y += x" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const n: usize = vl * 3 + 1;

    var rng = fx.FixtureRng.init(0xA8A8_0001);
    const r = rng.random();

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);

    fx.fillSliceRandom(r, x);
    fx.fillSliceRandom(r, y);

    const expected = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(expected);
    for (0..n) |i| expected[i] = x[i] + expected[i];

    blazt.ops.axpy(f32, n, @as(f32, 1.0), x, y);
    try fx.expectSliceApproxEq(f32, expected, y, fx.defaultTolerance(f32));
}

test "ops.axpy alpha=random matches scalar reference (incl tail)" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const len: usize = vl * 3 + 1;
    const n: usize = len - 2; // leave an untouched tail too

    var rng = fx.FixtureRng.init(0xA8A8_0002);
    const r = rng.random();

    const x = try std.testing.allocator.alloc(f32, len);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, len);
    defer std.testing.allocator.free(y);

    fx.fillSliceRandom(r, x);
    fx.fillSliceRandom(r, y);

    const expected = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(expected);

    const alpha = fx.randomScalar(r, f32);
    for (0..n) |i| expected[i] = alpha * x[i] + expected[i];
    // expected[n..] unchanged

    blazt.ops.axpy(f32, n, alpha, x, y);
    try fx.expectSliceApproxEq(f32, expected, y, fx.defaultTolerance(f32));
}


