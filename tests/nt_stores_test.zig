const std = @import("std");
const blazt = @import("blazt");
const build_options = blazt.build_options;

test "ops.copy non-temporal store path (when enabled) is correct" {
    if (!comptime build_options.nt_stores) return error.SkipZigTest;

    const n: usize = @max(@as(usize, 1) << 20, (build_options.nt_store_min_bytes + 63) / 64 * 64);
    const x = try blazt.allocAligned(std.testing.allocator, u8, n);
    defer std.testing.allocator.free(x);
    const y = try blazt.allocAligned(std.testing.allocator, u8, n);
    defer std.testing.allocator.free(y);

    // Deterministic pattern.
    for (x, 0..) |*v, i| v.* = @intCast(i % 251);
    @memset(y, 0xAA);

    blazt.ops.copy(u8, n, x, y);
    try std.testing.expectEqualSlices(u8, x, y);
}

test "memory.memsetZeroBytes (when enabled) zeros correctly" {
    if (!comptime build_options.nt_stores) return error.SkipZigTest;

    const n: usize = @max(build_options.nt_store_min_bytes, 1 << 20);
    const buf = try blazt.allocAligned(std.testing.allocator, u8, n);
    defer std.testing.allocator.free(buf);

    @memset(buf, 0xAA);
    blazt.memory.memsetZeroBytes(buf);

    for (buf) |v| try std.testing.expectEqual(@as(u8, 0), v);
}

test "ops.copy falls back safely when non-temporal stores cannot be used" {
    // Even if enabled, misalignment should fall back and still be correct.
    if (!comptime build_options.nt_stores) return error.SkipZigTest;

    var buf: [1024 + 1]u8 = undefined;
    for (&buf, 0..) |*v, i| v.* = @intCast(i % 251);

    const src = buf[0..1024];
    const dst = buf[1..];
    const expected_dst: [512]u8 = src[0..512].*;

    // Overlap + misalignment forces fallback behavior.
    blazt.ops.copy(u8, 512, src[0..512], dst[0..512]);
    try std.testing.expectEqualSlices(u8, expected_dst[0..], dst[0..512]);
}


