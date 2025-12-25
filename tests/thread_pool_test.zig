const builtin = @import("builtin");
const std = @import("std");
const blazt = @import("blazt");

fn incr(ctx: *anyopaque) void {
    const counter: *std.atomic.Value(u32) = @ptrCast(@alignCast(ctx));
    _ = counter.fetchAdd(1, .acq_rel);
}

const GateCtx = struct {
    gate: std.atomic.Value(u32) = .init(0),
    counter: std.atomic.Value(u32) = .init(0),
};

fn blockedIncr(ctx: *anyopaque) void {
    const g: *GateCtx = @ptrCast(@alignCast(ctx));
    while (g.gate.load(.acquire) == 0) {
        std.Thread.Futex.wait(&g.gate, 0);
    }
    _ = g.counter.fetchAdd(1, .acq_rel);
}

test "ThreadPool executes submitted work and waitAll completes" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var pool = try blazt.ThreadPool.init(std.testing.allocator, .{
        .thread_count = 4,
        .task_capacity = 4096,
        .deque_capacity = 4096,
    });
    defer pool.deinit();

    var counter = std.atomic.Value(u32).init(0);

    const n: u32 = 1000;
    for (0..n) |_| {
        try pool.submit(incr, &counter);
    }

    pool.waitAll();
    try std.testing.expectEqual(n, counter.load(.acquire));
}

test "ThreadPool supports multiple producer threads calling submit" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var pool = try blazt.ThreadPool.init(std.testing.allocator, .{
        .thread_count = 4,
        .task_capacity = 4096,
        .deque_capacity = 4096,
    });
    defer pool.deinit();

    var counter = std.atomic.Value(u32).init(0);

    const Producer = struct {
        pool: *blazt.ThreadPool,
        counter: *std.atomic.Value(u32),
        n: u32,
        fn run(self: *@This()) void {
            for (0..self.n) |_| {
                // Ignore AtCapacity here; capacity is sized to avoid it.
                _ = self.pool.submit(incr, self.counter) catch {};
            }
        }
    };

    var producers: [4]std.Thread = undefined;
    var ctx = Producer{ .pool = &pool, .counter = &counter, .n = 500 };
    for (&producers) |*t| t.* = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    for (producers) |t| t.join();

    pool.waitAll();
    try std.testing.expectEqual(@as(u32, 4) * ctx.n, counter.load(.acquire));
}

test "ThreadPool returns error.AtCapacity when task pool is exhausted" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var pool = try blazt.ThreadPool.init(std.testing.allocator, .{
        .thread_count = 2,
        .task_capacity = 8,
        .deque_capacity = 8,
    });
    defer pool.deinit();

    // Deterministically exhaust capacity by submitting tasks that block until released.
    var ctx: GateCtx = .{};

    for (0..8) |_| {
        try pool.submit(blockedIncr, &ctx);
    }

    try std.testing.expectError(error.AtCapacity, pool.submit(blockedIncr, &ctx));

    // Release tasks and ensure they all completed.
    ctx.gate.store(1, .release);
    std.Thread.Futex.wake(&ctx.gate, std.math.maxInt(u32));
    pool.waitAll();
    try std.testing.expectEqual(@as(u32, 8), ctx.counter.load(.acquire));
}


