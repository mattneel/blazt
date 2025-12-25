const std = @import("std");
const blazt = @import("blazt");

test "allocAligned returns CacheLine-aligned storage" {
    const allocator = std.testing.allocator;
    const buf = try blazt.allocAligned(allocator, u8, blazt.CacheLine * 2);
    defer allocator.free(buf);

    try std.testing.expect(@intFromPtr(buf.ptr) % blazt.CacheLine == 0);
}

test "ensureAligned returns a pointer with stronger alignment" {
    const allocator = std.testing.allocator;
    const buf = try blazt.allocAligned(allocator, u8, blazt.CacheLine);
    defer allocator.free(buf);

    const p = blazt.ensureAligned(blazt.CacheLine, buf.ptr);
    const alignment: usize = @intCast(@typeInfo(@TypeOf(p)).pointer.alignment);
    try std.testing.expect(alignment >= blazt.CacheLine);
    try std.testing.expect(@intFromPtr(p) % blazt.CacheLine == 0);
}


