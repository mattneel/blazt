const builtin = @import("builtin");
const std = @import("std");
const blazt = @import("blazt");

test "CpuInfo.native mirrors builtin feature detection" {
    const info = blazt.CpuInfo.native();

    try std.testing.expectEqual(builtin.cpu.has(.x86, .avx512f), info.has_avx512f);
    try std.testing.expectEqual(builtin.cpu.has(.x86, .avx2), info.has_avx2);
    try std.testing.expectEqual(builtin.cpu.has(.x86, .fma), info.has_fma);
    try std.testing.expectEqual(builtin.cpu.has(.arm, .neon) or builtin.cpu.has(.aarch64, .neon), info.has_neon);
    try std.testing.expectEqual(builtin.cpu.has(.aarch64, .sve), info.has_sve);
}

test "CpuInfo.native uses conservative cache defaults (Zig 0.16 has no builtin cpu.cache)" {
    const info = blazt.CpuInfo.native();
    const expected_l1d = blazt.cpu.CacheDefaults.l1d_size_bytes;
    const expected_line = blazt.cpu.CacheDefaults.l1d_line_bytes;
    const expected_l2 = blazt.cpu.CacheDefaults.l2_size_bytes;
    const expected_l3 = blazt.cpu.CacheDefaults.l3_size_bytes;

    try std.testing.expectEqual(expected_l1d, info.l1d_size_bytes);
    try std.testing.expectEqual(expected_line, info.l1d_line_bytes);
    try std.testing.expectEqual(expected_l2, info.l2_size_bytes);
    try std.testing.expectEqual(expected_l3, info.l3_size_bytes);

    try std.testing.expect(info.l1d_size_bytes > 0);
    try std.testing.expect(info.l1d_line_bytes > 0);
    try std.testing.expect(info.l2_size_bytes > 0);
    try std.testing.expect(info.l3_size_bytes > 0);
}


