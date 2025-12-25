const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

test "ops.copy copies n elements (various sizes)" {
    var rng = fx.FixtureRng.init(0xC0FFEE);
    const r = rng.random();

    inline for (.{ 0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128 }) |n| {
        const x = try std.testing.allocator.alloc(u64, n);
        defer std.testing.allocator.free(x);
        const y = try std.testing.allocator.alloc(u64, n + 3);
        defer std.testing.allocator.free(y);

        // Fill x with deterministic random data; y with sentinel.
        for (x) |*v| v.* = r.int(u64);
        @memset(y, 0xDEADBEEF);

        blazt.ops.copy(u64, n, x, y);

        try std.testing.expectEqualSlices(u64, x, y[0..n]);
        if (y.len > n) {
            // ensure we didn't touch beyond n
            for (y[n..]) |v| try std.testing.expectEqual(@as(u64, 0xDEADBEEF), v);
        }
    }
}

test "ops.copy is overlap-safe (memmove semantics)" {
    // dst starts after src (requires backward copy)
    {
        var buf: [16]u8 = undefined;
        for (&buf, 0..) |*v, i| v.* = @intCast(i);

        const x = buf[0..10];
        const y = buf[1..11];
        blazt.ops.copy(u8, 10, x, y);

        // expected: buf[1..11] = 0..9
        try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15 }, &buf);
    }

    // dst starts before src (forward copy is safe)
    {
        var buf: [16]u8 = undefined;
        for (&buf, 0..) |*v, i| v.* = @intCast(i);

        const x = buf[1..11];
        const y = buf[0..10];
        blazt.ops.copy(u8, 10, x, y);

        try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15 }, &buf);
    }
}


