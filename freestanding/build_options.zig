// Build options for freestanding RISC-V target.
// Disable x86-specific features.
pub const gemm_prefetch: bool = false; // No prefetch on freestanding
pub const gemm_prefetch_a_k: usize = 0;
pub const gemm_prefetch_b_k: usize = 0;
pub const gemm_prefetch_locality: u2 = 0;
pub const nt_stores: bool = false; // No NT stores on RISC-V
pub const nt_store_min_bytes: usize = 0;
