const builtin = @import("builtin");
const std = @import("std");

pub const cache = @import("cpu_cache");

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
    // Cache sharing (logical CPUs per cache instance)
    l1d_shared_by_logical_cpus: usize,
    l2_shared_by_logical_cpus: usize,
    l3_shared_by_logical_cpus: usize,

    pub fn native() CpuInfo {
        const cpu = builtin.cpu;

        // Zig 0.16: Target.Cpu does not expose cache sizes; use build-time-probed cache
        // constants when available (see `build.zig`), otherwise fall back to defaults.
        const l1d_size = cache.l1d_size_bytes;
        const l1d_line = cache.l1d_line_bytes;
        const l2_size = cache.l2_size_bytes;
        const l3_size = cache.l3_size_bytes;
        const l1d_shared = cache.l1d_shared_by_logical_cpus;
        const l2_shared = cache.l2_shared_by_logical_cpus;
        const l3_shared = cache.l3_shared_by_logical_cpus;

        std.debug.assert(l1d_size > 0);
        std.debug.assert(l1d_line > 0);
        std.debug.assert(l2_size > 0);
        std.debug.assert(l3_size > 0);
        std.debug.assert(l1d_shared > 0);
        std.debug.assert(l2_shared > 0);
        std.debug.assert(l3_shared > 0);

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
            .l1d_shared_by_logical_cpus = l1d_shared,
            .l2_shared_by_logical_cpus = l2_shared,
            .l3_shared_by_logical_cpus = l3_shared,
        };
    }
};
