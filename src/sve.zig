const builtin = @import("builtin");
const std = @import("std");

/// SVE support helpers.
///
/// Note: Zig `@Vector` types are fixed-length at comptime. SVE's vector length (VL) is runtime
/// configurable. This module provides a runtime VL probe (when compiled for aarch64+sve) plus a
/// small dispatch helper to select from fixed-width kernels.
pub const sve = struct {
    /// True if the compilation target has the SVE feature enabled.
    pub const has_sve: bool = builtin.cpu.arch == .aarch64 and builtin.cpu.has(.aarch64, .sve);

    /// Runtime SVE vector length in **bytes**.
    ///
    /// On non-aarch64/non-SVE targets, returns 0.
    pub fn runtimeVlBytes() usize {
        if (!comptime has_sve) return 0;

        // Inline asm for SVE is not available under the C backend.
        if (comptime builtin.zig_backend == .stage2_c) return 0;

        var out: usize = 0;
        // RDVL: Read Vector Length (bytes). Immediate multiplies VL; we use 1.
        asm volatile ("rdvl %[out], #1"
            : [out] "=&r" (out),
        );
        return out;
    }

    /// Runtime SVE vector length in elements of `T` (0 if unknown/unavailable).
    pub fn runtimeVlElems(comptime T: type) usize {
        const b = runtimeVlBytes();
        if (b == 0) return 0;
        return b / @sizeOf(T);
    }

    /// Dispatch helper: selects the first candidate `VL` (elements) that fits in `runtime_vl_elems`.
    ///
    /// - `candidates_desc` must be a comptime-known list of candidate vector lengths (elements),
    ///   in **descending** order (largest first).
    /// - Calls `kernel(VL, ctx)` where `VL` is comptime-known.
    ///
    /// Returns true if a candidate was selected; false if it fell back to scalar (VL=1).
    pub fn dispatchFixedVectorLen(
        comptime candidates_desc: []const usize,
        runtime_vl_elems: usize,
        kernel: anytype,
        ctx: anytype,
    ) bool {
        comptime std.debug.assert(candidates_desc.len > 0);

        inline for (candidates_desc) |VL| {
            comptime std.debug.assert(VL > 0);
            if (runtime_vl_elems >= VL) {
                kernel(VL, ctx);
                return true;
            }
        }

        kernel(1, ctx);
        return false;
    }
};


