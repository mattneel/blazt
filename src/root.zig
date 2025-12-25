//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const memory = @import("memory.zig");
pub const CacheLine = memory.CacheLine;
pub const allocAligned = memory.allocAligned;
pub const ensureAligned = memory.ensureAligned;

pub const matrix = @import("matrix.zig");
pub const Layout = matrix.Layout;
pub const Matrix = matrix.Matrix;

pub const types = @import("types.zig");
pub const Trans = types.Trans;
pub const UpLo = types.UpLo;
pub const Side = types.Side;
pub const Diag = types.Diag;

pub const errors = @import("errors.zig");
pub const LuError = errors.LuError;
pub const CholeskyError = errors.CholeskyError;
pub const TrsvError = errors.TrsvError;

pub const ops = @import("ops.zig").ops;
pub const parallel = @import("parallel.zig").parallel;

pub const cpu = @import("cpu.zig");
pub const CpuInfo = cpu.CpuInfo;

pub const simd = @import("simd.zig");
pub const sve = @import("sve.zig").sve;

pub const gemm = @import("gemm.zig");

pub const bench = @import("bench.zig");
pub const oracle = @import("oracle.zig");

/// Generated build-time configuration knobs (from `build.zig`).
pub const build_options = @import("build_options");

pub const thread_pool = @import("thread_pool.zig");
pub const ThreadPool = thread_pool.ThreadPool;

// Avoid name ambiguity inside `GemmBuilder` methods (which include a method named `parallel`).
const parallel_mod = parallel;

/// Builder-style configuration for GEMM.
///
/// Note: this is intentionally named `gemmBuilder` (not `gemm`) because `blazt.gemm`
/// is already used to expose the GEMM kernel module (`src/gemm.zig`).
pub fn gemmBuilder(comptime T: type) GemmBuilder(T) {
    if (comptime @typeInfo(T) != .float) {
        @compileError("gemmBuilder only supports floating-point types");
    }
    return .{};
}

pub fn GemmBuilder(comptime T: type) type {
    return struct {
        const Self = @This();

        trans_a: Trans = .no_trans,
        trans_b: Trans = .no_trans,
        alpha_value: T = @as(T, 1),
        beta_value: T = @as(T, 0),
        pool: ?*ThreadPool = null,

        pub fn transA(self: Self, t: Trans) Self {
            var out = self;
            out.trans_a = t;
            return out;
        }

        pub fn transB(self: Self, t: Trans) Self {
            var out = self;
            out.trans_b = t;
            return out;
        }

        pub fn alpha(self: Self, a: T) Self {
            var out = self;
            out.alpha_value = a;
            return out;
        }

        pub fn beta(self: Self, b: T) Self {
            var out = self;
            out.beta_value = b;
            return out;
        }

        pub fn parallel(self: Self, thread_pool_ptr: *ThreadPool) Self {
            var out = self;
            out.pool = thread_pool_ptr;
            return out;
        }

        pub fn execute(self: Self, a: anytype, b: anytype, c: anytype) void {
            const A = @TypeOf(a);
            const B = @TypeOf(b);
            const CPtr = @TypeOf(c);

            comptime {
                if (!@hasDecl(A, "layout_const")) @compileError("execute(a,b,c): a must be blazt.Matrix(T, layout)");
                const layout = A.layout_const;

                if (A != Matrix(T, layout)) @compileError("execute(a,b,c): a must be blazt.Matrix(T, layout)");
                if (B != Matrix(T, layout)) @compileError("execute(a,b,c): b must be blazt.Matrix(T, layout)");

                if (@typeInfo(CPtr) != .pointer) @compileError("execute(a,b,c): c must be *blazt.Matrix(T, layout)");
                const p = @typeInfo(CPtr).pointer;
                if (p.size != .one) @compileError("execute(a,b,c): c must be *blazt.Matrix(T, layout)");
                if (p.child != Matrix(T, layout)) @compileError("execute(a,b,c): c must be *blazt.Matrix(T, layout)");
            }

            const layout = A.layout_const;

            if (layout == .row_major) {
                if (self.pool) |pool_ptr| {
                    parallel_mod.gemm(T, self.trans_a, self.trans_b, self.alpha_value, a, b, self.beta_value, c, pool_ptr);
                    return;
                }
            }

            ops.gemm(T, layout, self.trans_a, self.trans_b, self.alpha_value, a, b, self.beta_value, c);
        }
    };
}

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    // use std.log for logging
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

// re-export all public declarations
