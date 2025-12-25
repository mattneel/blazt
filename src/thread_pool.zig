const std = @import("std");

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,

    threads: []std.Thread,
    workers: []Worker,

    shared: *Shared,

    pub const InitOptions = struct {
        thread_count: usize = 0, // 0 => auto
        task_capacity: usize = 4096,
        deque_capacity: usize = 4096, // must be power of two
    };

    pub const SubmitError = error{
        AtCapacity,
        QueueFull,
    };

    pub fn init(allocator: std.mem.Allocator, options: InitOptions) !ThreadPool {
        const thread_count = if (options.thread_count != 0)
            options.thread_count
        else
            (std.Thread.getCpuCount() catch 4);

        std.debug.assert(thread_count > 0);
        std.debug.assert(options.task_capacity > 0);
        std.debug.assert(options.deque_capacity > 0);
        std.debug.assert(std.math.isPowerOfTwo(options.deque_capacity));
        std.debug.assert(options.deque_capacity >= options.task_capacity);

        const shared = try allocator.create(Shared);
        errdefer allocator.destroy(shared);

        shared.* = undefined;
        shared.inject.init();
        shared.deque = try WorkStealingDeque.init(allocator, options.deque_capacity);
        errdefer shared.deque.deinit(allocator);
        shared.task_pool = try TaskPool.init(allocator, options.task_capacity);
        errdefer shared.task_pool.deinit(allocator);
        shared.epoch = .init(0);
        shared.pending = .init(0);
        shared.shutdown = .init(false);
        errdefer shared.deinit(allocator);

        const threads = try allocator.alloc(std.Thread, thread_count);
        errdefer allocator.free(threads);

        const workers = try allocator.alloc(Worker, thread_count);
        errdefer allocator.free(workers);

        // Initialize worker contexts.
        for (workers, 0..) |*w, i| {
            w.* = .{
                .shared = shared,
                .index = @intCast(i),
            };
        }

        // Spawn workers.
        for (threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{}, Worker.main, .{&workers[i]});
        }

        return .{
            .allocator = allocator,
            .threads = threads,
            .workers = workers,
            .shared = shared,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        // Ensure all submitted work is complete before shutting down.
        self.waitAll();

        self.shared.shutdown.store(true, .release);
        self.signalWorkers();

        for (self.threads) |t| t.join();

        self.shared.deinit(self.allocator);
        self.allocator.destroy(self.shared);

        self.allocator.free(self.workers);
        self.allocator.free(self.threads);

        self.* = undefined;
    }

    pub fn submit(self: *ThreadPool, func: *const fn (*anyopaque) void, ctx: *anyopaque) SubmitError!void {
        const task = self.shared.task_pool.alloc() orelse return error.AtCapacity;
        task.func = func;
        task.ctx = ctx;

        // Track work before publishing the task.
        _ = self.shared.pending.fetchAdd(1, .acq_rel);

        self.shared.inject.push(task);
        self.signalWorkers();
    }

    pub fn waitAll(self: *ThreadPool) void {
        while (true) {
            const p = self.shared.pending.load(.seq_cst);
            if (p == 0) return;
            std.Thread.Futex.wait(&self.shared.pending, p);
        }
    }

    fn signalWorkers(self: *ThreadPool) void {
        // Bump epoch and wake everyone (cheap when nobody is sleeping; Futex is cold).
        _ = self.shared.epoch.fetchAdd(1, .seq_cst);
        std.Thread.Futex.wake(&self.shared.epoch, std.math.maxInt(u32));
    }
};

const Worker = struct {
    shared: *Shared,
    index: u32,

    fn main(self: *Worker) void {
        // Worker event loop (intentionally unbounded).
        while (true) {
            if (self.shared.shutdown.load(.acquire)) return;

            // Worker 0 drains the injection queue and owns push/pop on the deque.
            if (self.index == 0) {
                self.drainInject();
                if (self.shared.deque.pop()) |task| {
                    self.execute(task);
                    continue;
                }
            } else {
                if (self.shared.deque.steal()) |task| {
                    self.execute(task);
                    continue;
                }
            }

            // Nothing to do: sleep until epoch changes (or spurious wakeup).
            const expect = self.shared.epoch.load(.seq_cst);
            std.Thread.Futex.wait(&self.shared.epoch, expect);
        }
    }

    fn execute(self: *Worker, task: *Task) void {
        // Run user code.
        task.func(task.ctx);

        // Return task node to pool.
        self.shared.task_pool.free(task);

        // Mark completion and wake waiters if we hit zero.
        const old = self.shared.pending.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        if (old == 1) {
            std.Thread.Futex.wake(&self.shared.pending, std.math.maxInt(u32));
        }
    }

    fn drainInject(self: *Worker) void {
        while (self.shared.inject.pop()) |task| {
            self.shared.deque.push(task) catch {
                // Deque should be sized to hold all in-flight tasks. If not, fail loudly.
                @panic("thread pool deque overflow");
            };
        }
    }
};

const Task = struct {
    // Used by InjectQueue and TaskPool freelist.
    next: std.atomic.Value(?*Task) = .init(null),

    func: *const fn (*anyopaque) void = undefined,
    ctx: *anyopaque = undefined,
};

const TaskPool = struct {
    free_head: std.atomic.Value(?*Task),
    storage: []Task,

    fn init(allocator: std.mem.Allocator, capacity: usize) !TaskPool {
        var storage = try allocator.alloc(Task, capacity);
        // Build a simple singly-linked free list (no concurrency during init).
        for (storage[0..capacity - 1], 0..) |*t, i| {
            t.next.store(&storage[i + 1], .unordered);
        }
        storage[capacity - 1].next.store(null, .unordered);

        return .{
            .free_head = .init(&storage[0]),
            .storage = storage,
        };
    }

    fn deinit(self: *TaskPool, allocator: std.mem.Allocator) void {
        allocator.free(self.storage);
        self.* = undefined;
    }

    fn alloc(self: *TaskPool) ?*Task {
        while (true) {
            const head = self.free_head.load(.acquire) orelse return null;
            const next = head.next.load(.acquire);
            if (self.free_head.cmpxchgWeak(head, next, .acq_rel, .acquire) == null) {
                head.next.store(null, .unordered);
                return head;
            }
        }
    }

    fn free(self: *TaskPool, task: *Task) void {
        while (true) {
            const head = self.free_head.load(.acquire);
            task.next.store(head, .unordered);
            if (self.free_head.cmpxchgWeak(head, task, .acq_rel, .acquire) == null) return;
        }
    }
};

const InjectQueue = struct {
    head: std.atomic.Value(*Task),
    tail: *Task, // consumer-only
    stub: Task,

    fn init(self: *InjectQueue) void {
        self.stub = .{};
        self.head = .init(&self.stub);
        self.tail = &self.stub;
    }

    fn push(self: *InjectQueue, node: *Task) void {
        node.next.store(null, .monotonic);

        const prev = self.head.swap(node, .acq_rel);
        prev.next.store(node, .release);
    }

    fn pop(self: *InjectQueue) ?*Task {
        var tail = self.tail;
        var next = tail.next.load(.acquire);

        if (tail == &self.stub) {
            if (next == null) return null;
            self.tail = next.?;
            tail = next.?;
            next = tail.next.load(.acquire);
        }

        if (next) |n| {
            self.tail = n;
            return tail;
        }

        const head = self.head.load(.acquire);
        if (tail != head) return null;

        self.push(&self.stub);

        next = tail.next.load(.acquire);
        if (next) |n| {
            self.tail = n;
            return tail;
        }

        return null;
    }
};

const Shared = struct {
    inject: InjectQueue,
    deque: WorkStealingDeque,
    task_pool: TaskPool,
    epoch: std.atomic.Value(u32),
    pending: std.atomic.Value(u32),
    shutdown: std.atomic.Value(bool),

    fn deinit(self: *Shared, allocator: std.mem.Allocator) void {
        self.deque.deinit(allocator);
        self.task_pool.deinit(allocator);
        self.* = undefined;
    }
};

const WorkStealingDeque = struct {
    head: std.atomic.Value(u64) = .init(0),
    tail: std.atomic.Value(u64) = .init(0),
    buffer: []std.atomic.Value(?*Task),
    mask: u64,

    const Error = error{QueueFull};

    fn init(allocator: std.mem.Allocator, capacity: usize) !WorkStealingDeque {
        std.debug.assert(capacity > 0);
        std.debug.assert(std.math.isPowerOfTwo(capacity));

        const buffer = try allocator.alloc(std.atomic.Value(?*Task), capacity);
        for (buffer) |*slot| slot.* = .init(null);

        return .{
            .buffer = buffer,
            .mask = @intCast(capacity - 1),
        };
    }

    fn deinit(self: *WorkStealingDeque, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
        self.* = undefined;
    }

    fn push(self: *WorkStealingDeque, task: *Task) Error!void {
        const tail = self.tail.load(.monotonic);
        const head = self.head.load(.acquire);

        if (tail -% head >= self.buffer.len) {
            return error.QueueFull;
        }

        self.buffer[tail & self.mask].store(task, .unordered);
        self.tail.store(tail +% 1, .release);
    }

    fn pop(self: *WorkStealingDeque) ?*Task {
        var tail = self.tail.load(.monotonic);
        if (tail == 0) return null;

        tail -%= 1;
        // Zig 0.16: no std.atomic.fence; use seq_cst ops for the required ordering here.
        self.tail.store(tail, .seq_cst);
        const head = self.head.load(.seq_cst);

        if (head > tail) {
            self.tail.store(head, .monotonic);
            return null;
        }

        const task = self.buffer[tail & self.mask].load(.unordered);

        if (head == tail) {
            // Last item: race with thieves.
            if (self.head.cmpxchgStrong(head, head +% 1, .seq_cst, .monotonic)) |_| {
                self.tail.store(head +% 1, .monotonic);
                return null;
            }
            self.tail.store(head +% 1, .monotonic);
        }

        return task;
    }

    fn steal(self: *WorkStealingDeque) ?*Task {
        var head = self.head.load(.acquire);

        while (true) {
            const tail = self.tail.load(.acquire);

            if (head >= tail) return null;

            const task = self.buffer[head & self.mask].load(.unordered);

            if (self.head.cmpxchgWeak(head, head +% 1, .seq_cst, .monotonic)) |new_head| {
                head = new_head;
                continue;
            }
            return task;
        }
    }
};


