const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

test "ops.swap swaps n elements and preserves tails" {
    var rng = fx.FixtureRng.init(0x5A5A_0001);
    const r = rng.random();

    inline for (.{ 0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128 }) |n| {
        const x = try std.testing.allocator.alloc(u64, n + 3);
        defer std.testing.allocator.free(x);
        const y = try std.testing.allocator.alloc(u64, n + 5);
        defer std.testing.allocator.free(y);

        for (x) |*v| v.* = r.int(u64);
        for (y) |*v| v.* = r.int(u64);

        const x_before = try std.testing.allocator.dupe(u64, x);
        defer std.testing.allocator.free(x_before);
        const y_before = try std.testing.allocator.dupe(u64, y);
        defer std.testing.allocator.free(y_before);

        blazt.ops.swap(u64, n, x, y);

        try std.testing.expectEqualSlices(u64, y_before[0..n], x[0..n]);
        try std.testing.expectEqualSlices(u64, x_before[0..n], y[0..n]);

        // tails untouched
        try std.testing.expectEqualSlices(u64, x_before[n..], x[n..]);
        try std.testing.expectEqualSlices(u64, y_before[n..], y[n..]);
    }
}

test "ops.swap handles odd sizes (SIMD tail) correctly (f32)" {
    const vl: usize = @intCast(blazt.simd.suggestVectorLength(f32) orelse 1);
    const n: usize = vl * 3 + 1; // odd / forces scalar tail when vl>1

    var rng = fx.FixtureRng.init(0x5A5A_0002);
    const r = rng.random();

    const x = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(y);

    fx.fillSliceRandom(r, x);
    fx.fillSliceRandom(r, y);

    const x_before = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(x_before);
    const y_before = try std.testing.allocator.dupe(f32, y);
    defer std.testing.allocator.free(y_before);

    blazt.ops.swap(f32, n, x, y);

    try std.testing.expectEqualSlices(f32, y_before, x);
    try std.testing.expectEqualSlices(f32, x_before, y);
}

test "ops.swap is overlap-safe (no aliasing bugs)" {
    var buf: [16]u8 = undefined;
    for (&buf, 0..) |*v, i| v.* = @intCast(i);

    // Overlapping ranges.
    const n: usize = 10;
    const x = buf[0..n];
    const y = buf[1 .. 1 + n];

    // Compute expected result by applying the scalar swap semantics.
    var expected = buf;
    for (0..n) |i| {
        const tmp = expected[i];
        expected[i] = expected[i + 1];
        expected[i + 1] = tmp;
    }

    blazt.ops.swap(u8, n, x, y);
    try std.testing.expectEqualSlices(u8, &expected, &buf);
}


