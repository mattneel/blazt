// Tensix/RISC-V freestanding cache defaults.
// Tensix cores have scratchpad memory, not traditional cache hierarchy.
// These values are tuning hints for blocking - can be adjusted once running on hardware.
pub const detected: bool = false;
pub const method: []const u8 = "freestanding-tensix";

pub const l1d_size_bytes: usize = 16 * 1024; // 16KB conservative default
pub const l1d_line_bytes: usize = 64; // Standard cache line
// GEMM tiling requires non-zero L2/L3 for comptime assertions.
// Use L1 size as a conservative "virtual L2" for single-core operation.
pub const l2_size_bytes: usize = 64 * 1024; // 64KB virtual L2 for tiling
pub const l3_size_bytes: usize = 256 * 1024; // 256KB virtual L3 for tiling

pub const l1d_shared_by_logical_cpus: usize = 1;
pub const l2_shared_by_logical_cpus: usize = 1;
pub const l3_shared_by_logical_cpus: usize = 1;
