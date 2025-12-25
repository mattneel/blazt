// Root test aggregator, tests are colocated in their own folder in this project
const std = @import("std");

comptime {
    _ = @import("memory_test.zig");
    _ = @import("matrix_test.zig");
    _ = @import("cpu_test.zig");
    _ = @import("simd_test.zig");
    _ = @import("fixtures_test.zig");
    _ = @import("types_errors_test.zig");
    _ = @import("thread_pool_test.zig");
    _ = @import("bench_test.zig");
    _ = @import("oracle_test.zig");
    _ = @import("copy_test.zig");
    _ = @import("scal_test.zig");
    _ = @import("swap_test.zig");
    _ = @import("axpy_test.zig");
    _ = @import("dot_test.zig");
    _ = @import("nrm2_asum_test.zig");
    _ = @import("iamax_iamin_test.zig");
    _ = @import("blas1_oracle_parity_test.zig");
    _ = @import("gemv_test.zig");
    _ = @import("ger_test.zig");
    _ = @import("trmv_trsv_test.zig");
    _ = @import("symv_hemv_test.zig");
    _ = @import("blas2_oracle_parity_test.zig");
    _ = @import("gemm_tile_params_test.zig");
    _ = @import("gemm_microkernel_test.zig");
    _ = @import("gemm_pack_test.zig");
    _ = @import("gemm_macrokernel_test.zig");
    _ = @import("gemm_kernel_variant_test.zig");
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
