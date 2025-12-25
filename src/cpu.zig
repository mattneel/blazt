const builtin = @import("builtin");
const std = @import("std");

pub const CacheDefaults = struct {
    // Conservative defaults when target cache info is unknown.
    pub const l1d_size_bytes: usize = 32 * 1024;
    // Prefer std.atomic.cache_line when available (Zig 0.16 does not expose cpu.cache).
    pub const l1d_line_bytes: usize = std.atomic.cache_line;
    pub const l2_size_bytes: usize = 256 * 1024;
    pub const l3_size_bytes: usize = 8 * 1024 * 1024;
};

pub const CpuInfo = struct {
    // Feature flags (comptime-known for the target)
    has_avx512f: bool,
    has_avx2: bool,
    has_fma: bool,
    has_neon: bool,
    has_sve: bool,

    // Cache hierarchy (bytes)
    l1d_size_bytes: usize,
    l1d_line_bytes: usize,
    l2_size_bytes: usize,
    l3_size_bytes: usize,

    pub fn native() CpuInfo {
        const cpu = builtin.cpu;

        // Zig 0.16: Target.Cpu does not expose cache sizes; use conservative defaults.
        const l1d_size = CacheDefaults.l1d_size_bytes;
        const l1d_line = CacheDefaults.l1d_line_bytes;
        const l2_size = CacheDefaults.l2_size_bytes;
        const l3_size = CacheDefaults.l3_size_bytes;

        std.debug.assert(l1d_size > 0);
        std.debug.assert(l1d_line > 0);
        std.debug.assert(l2_size > 0);
        std.debug.assert(l3_size > 0);

        return .{
            // Feature detection is arch-family aware in Zig 0.16.
            .has_avx512f = cpu.has(.x86, .avx512f),
            .has_avx2 = cpu.has(.x86, .avx2),
            .has_fma = cpu.has(.x86, .fma),
            .has_neon = cpu.has(.arm, .neon) or cpu.has(.aarch64, .neon),
            .has_sve = cpu.has(.aarch64, .sve),
            .l1d_size_bytes = l1d_size,
            .l1d_line_bytes = l1d_line,
            .l2_size_bytes = l2_size,
            .l3_size_bytes = l3_size,
        };
    }
};


