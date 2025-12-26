const std = @import("std");
const builtin = @import("builtin");

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,

    threads: []std.Thread,
    workers: []Worker,

    shared: *Shared,

    pub const InitOptions = struct {
        thread_count: usize = 0, // 0 => auto
        task_capacity: usize = 4096,
        deque_capacity: usize = 4096, // must be power of two
        /// When enabled, each worker thread pins itself to a specific CPU index to reduce
        /// scheduler noise during benchmarks.
        ///
        /// This is a best-effort hint: failures are ignored.
        pin_threads: bool = false,
        /// Base CPU index used when `pin_threads` is enabled.
        pin_base_cpu: usize = 0,
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
        shared.mutex = .{};
        shared.cond = .{};
        {
            const queue = try allocator.alloc(?*Task, options.deque_capacity);
            errdefer allocator.free(queue);
            @memset(queue, null);

            var task_pool = try TaskPool.init(allocator, options.task_capacity);
            errdefer task_pool.deinit(allocator);

            shared.queue = queue;
            shared.task_pool = task_pool;
        }
        shared.q_mask = options.deque_capacity - 1;
        shared.q_head = 0;
        shared.q_tail = 0;
        shared.q_len = 0;
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
                .pin_cpu = if (options.pin_threads) options.pin_base_cpu + i else null,
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

        self.shared.mutex.lock();
        self.shared.shutdown.store(true, .release);
        self.shared.cond.broadcast();
        self.shared.mutex.unlock();

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

        self.shared.mutex.lock();
        defer self.shared.mutex.unlock();

        if (self.shared.q_len >= self.shared.queue.len) {
            // Should not happen when queue capacity >= task pool capacity, but handle gracefully.
            _ = self.shared.pending.fetchSub(1, .acq_rel);
            self.shared.task_pool.free(task);
            return error.QueueFull;
        }

        self.shared.queue[self.shared.q_tail] = task;
        self.shared.q_tail = (self.shared.q_tail + 1) & self.shared.q_mask;
        self.shared.q_len += 1;
        self.shared.cond.signal();
    }

    pub fn waitAll(self: *ThreadPool) void {
        while (true) {
            const p = self.shared.pending.load(.seq_cst);
            if (p == 0) return;
            std.Thread.Futex.wait(&self.shared.pending, p);
        }
    }

    // Worker wakeups are managed via `shared.cond`.
};

const Worker = struct {
    shared: *Shared,
    index: u32,
    pin_cpu: ?usize,

    fn main(self: *Worker) void {
        if (self.pin_cpu) |cpu| {
            // Best-effort; ignore failures to keep the pool usable in restricted environments.
            pinCurrentThreadToCpu(cpu) catch {};
        }
        while (true) {
            self.shared.mutex.lock();
            while (self.shared.q_len == 0 and !self.shared.shutdown.load(.acquire)) {
                self.shared.cond.wait(&self.shared.mutex);
            }

            if (self.shared.q_len == 0 and self.shared.shutdown.load(.acquire)) {
                self.shared.mutex.unlock();
                return;
            }

            const task = self.shared.queue[self.shared.q_head].?;
            self.shared.queue[self.shared.q_head] = null;
            self.shared.q_head = (self.shared.q_head + 1) & self.shared.q_mask;
            self.shared.q_len -= 1;
            self.shared.mutex.unlock();

            self.execute(task);
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

    // (no per-worker queue helpers; work is taken from the shared ring buffer)
};

fn pinCurrentThreadToCpu(cpu_index: usize) !void {
    if (builtin.os.tag != .linux) return;

    // Linux `sched_setaffinity` takes a cpuset bitmask (up to 1024 CPUs with glibc layout).
    var set = std.mem.zeroes(std.os.linux.cpu_set_t);

    const bits_per_word: usize = @bitSizeOf(usize);
    const word: usize = cpu_index / bits_per_word;
    const bit: usize = cpu_index % bits_per_word;
    if (word >= set.len) return;
    set[word] |= (@as(usize, 1) << @intCast(bit));

    try std.os.linux.sched_setaffinity(0, &set);
}

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
        if (tail != head) {
            // A producer has advanced `head` but has not linked `tail.next` yet.
            // If we return null here, the worker may go to sleep and miss work.
            while (true) {
                next = tail.next.load(.acquire);
                if (next) |n| {
                    self.tail = n;
                    return tail;
                }
                std.atomic.spinLoopHint();
            }
        }

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
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    queue: []?*Task,
    q_mask: usize,
    q_head: usize,
    q_tail: usize,
    q_len: usize,
    task_pool: TaskPool,
    pending: std.atomic.Value(u32),
    shutdown: std.atomic.Value(bool),

    fn deinit(self: *Shared, allocator: std.mem.Allocator) void {
        allocator.free(self.queue);
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

        // Publish the task pointer before making it visible via `tail`.
        self.buffer[tail & self.mask].store(task, .release);
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

        // This slot must have been published before `tail` advanced.
        const task = self.buffer[tail & self.mask].load(.acquire) orelse blk: {
            // Extremely unlikely: if publication is observed out-of-order, spin until visible.
            while (true) {
                if (self.buffer[tail & self.mask].load(.acquire)) |t| break :blk t;
                std.atomic.spinLoopHint();
            }
        };

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

            if (self.head.cmpxchgWeak(head, head +% 1, .seq_cst, .monotonic)) |new_head| {
                head = new_head;
                continue;
            }

            // Claimed index `head` successfully; wait until the slot's pointer is visible.
            while (true) {
                if (self.buffer[head & self.mask].load(.acquire)) |task| return task;
                std.atomic.spinLoopHint();
            }
        }
    }
};


