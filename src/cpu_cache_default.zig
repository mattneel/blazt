const std = @import("std");

// Fallback cache constants when build-time probing is disabled or unavailable.
// These match `blazt.cpu.CacheDefaults`.
pub const detected: bool = false;
pub const method: []const u8 = "default";

pub const l1d_size_bytes: usize = 32 * 1024;
pub const l1d_line_bytes: usize = std.atomic.cache_line;
pub const l2_size_bytes: usize = 256 * 1024;
pub const l3_size_bytes: usize = 8 * 1024 * 1024;

// Cache sharing (logical CPUs per cache instance). Conservative defaults.
pub const l1d_shared_by_logical_cpus: usize = 1;
pub const l2_shared_by_logical_cpus: usize = 1;
pub const l3_shared_by_logical_cpus: usize = 1;
