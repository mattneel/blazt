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

pub const gemm = @import("gemm.zig");

pub const bench = @import("bench.zig");
pub const oracle = @import("oracle.zig");

pub const thread_pool = @import("thread_pool.zig");
pub const ThreadPool = thread_pool.ThreadPool;

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
