const std = @import("std");
const blazt = @import("blazt");

test "simd.reduce(.Add) sums lanes" {
    const V = @Vector(4, u32);
    const v: V = .{ 1, 2, 3, 4 };
    try std.testing.expectEqual(@as(u32, 10), blazt.simd.reduce(.Add, v));
}

test "simd.reverse reverses lanes" {
    const V = @Vector(4, u32);
    const v: V = .{ 1, 2, 3, 4 };
    const r = blazt.simd.reverse(v);
    try std.testing.expectEqual(V{ 4, 3, 2, 1 }, r);
}

test "simd.interleaveLower/Upper interleave halves" {
    const V = @Vector(4, u32);
    const a: V = .{ 0, 1, 2, 3 };
    const b: V = .{ 10, 11, 12, 13 };

    const lo = blazt.simd.interleaveLower(a, b);
    const hi = blazt.simd.interleaveUpper(a, b);

    try std.testing.expectEqual(V{ 0, 10, 1, 11 }, lo);
    try std.testing.expectEqual(V{ 2, 12, 3, 13 }, hi);
}

test "simd.prefetch compiles" {
    var buf: [64]u8 = undefined;
    blazt.simd.prefetch(&buf[0], .{ .rw = .read, .locality = 3, .cache = .data });
}


