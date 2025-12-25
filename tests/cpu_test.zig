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

test "CpuInfo.native uses build-time-probed caches (or defaults when probing is disabled)" {
    const info = blazt.CpuInfo.native();
    const expected_l1d = blazt.cpu.cache.l1d_size_bytes;
    const expected_line = blazt.cpu.cache.l1d_line_bytes;
    const expected_l2 = blazt.cpu.cache.l2_size_bytes;
    const expected_l3 = blazt.cpu.cache.l3_size_bytes;
    const expected_l1d_shared = blazt.cpu.cache.l1d_shared_by_logical_cpus;
    const expected_l2_shared = blazt.cpu.cache.l2_shared_by_logical_cpus;
    const expected_l3_shared = blazt.cpu.cache.l3_shared_by_logical_cpus;

    try std.testing.expectEqual(expected_l1d, info.l1d_size_bytes);
    try std.testing.expectEqual(expected_line, info.l1d_line_bytes);
    try std.testing.expectEqual(expected_l2, info.l2_size_bytes);
    try std.testing.expectEqual(expected_l3, info.l3_size_bytes);
    try std.testing.expectEqual(expected_l1d_shared, info.l1d_shared_by_logical_cpus);
    try std.testing.expectEqual(expected_l2_shared, info.l2_shared_by_logical_cpus);
    try std.testing.expectEqual(expected_l3_shared, info.l3_shared_by_logical_cpus);

    try std.testing.expect(info.l1d_size_bytes > 0);
    try std.testing.expect(info.l1d_line_bytes > 0);
    try std.testing.expect(info.l2_size_bytes > 0);
    try std.testing.expect(info.l3_size_bytes > 0);
    try std.testing.expect(info.l1d_shared_by_logical_cpus > 0);
    try std.testing.expect(info.l2_shared_by_logical_cpus > 0);
    try std.testing.expect(info.l3_shared_by_logical_cpus > 0);
}
