const std = @import("std");
const blazt = @import("blazt");

test "gemm.selectGemmKernel chooses expected variants (comptime)" {
    comptime {
        std.debug.assert(blazt.gemm.selectGemmKernel(f32, 1.0, 0.0) == .alpha_one_beta_zero);
        std.debug.assert(blazt.gemm.selectGemmKernel(f32, 1.0, 1.0) == .alpha_one_beta_one);
        std.debug.assert(blazt.gemm.selectGemmKernel(f32, 1.0, 2.0) == .alpha_one);
        std.debug.assert(blazt.gemm.selectGemmKernel(f32, 2.0, 0.0) == .beta_zero);
        std.debug.assert(blazt.gemm.selectGemmKernel(f32, 2.0, 1.0) == .beta_one);
        std.debug.assert(blazt.gemm.selectGemmKernel(f32, 2.0, 3.0) == .generic);

        // Use the result in a comptime switch to ensure it's comptime-known when alpha/beta are comptime.
        const v = blazt.gemm.selectGemmKernel(f32, 1.0, 0.0);
        _ = switch (v) {
            .alpha_one_beta_zero => true,
            else => @compileError("wrong variant"),
        };
    }
}


