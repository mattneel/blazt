const builtin = @import("builtin");
const std = @import("std");
const blazt = @import("blazt");

test "bench.percentileSortedNs / medianSortedNs" {
    const s = [_]u64{ 10, 20, 30, 40, 50 };
    try std.testing.expectEqual(@as(u64, 10), blazt.bench.percentileSortedNs(&s, 0.0));
    try std.testing.expectEqual(@as(u64, 50), blazt.bench.percentileSortedNs(&s, 1.0));
    try std.testing.expectEqual(@as(u64, 30), blazt.bench.medianSortedNs(&s));
    try std.testing.expectEqual(@as(u64, 40), blazt.bench.percentileSortedNs(&s, 0.9));
}

test "bench.gflops matches flops/ns identity" {
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), blazt.bench.gflops(2, 1), 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), blazt.bench.gflops(3, 2), 0.0);
}

fn tinyWork() void {
    var x: u64 = 0;
    for (0..10_000) |i| {
        x +%= i;
    }
    std.mem.doNotOptimizeAway(x);
}

test "bench.run returns requested sample count" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var res = blazt.bench.run(std.testing.allocator, "tinyWork", .{
        .warmup_iters = 1,
        .samples = 5,
        .inner_iters = 10,
    }, tinyWork, .{}) catch |err| switch (err) {
        error.TimerUnsupported => return error.SkipZigTest,
        else => return err,
    };
    defer res.deinit();

    try std.testing.expectEqual(@as(usize, 5), res.samples_ns.len);
}


