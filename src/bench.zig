const std = @import("std");

pub const RunOptions = struct {
    warmup_iters: usize = 10,
    samples: usize = 30,
    inner_iters: usize = 1,
};

pub const RunResult = struct {
    name: []const u8,
    samples_ns: []u64,
    /// Iterations executed per collected sample (used to amortize timer overhead).
    iters_per_sample: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RunResult) void {
        self.allocator.free(self.samples_ns);
        self.* = undefined;
    }

    pub fn sortInPlace(self: *RunResult) void {
        sortNs(self.samples_ns);
    }
};

/// Run a benchmark for a callable `func` with arguments `args`.
///
/// Returns per-iteration nanoseconds (each sample measures `inner_iters` calls and is divided by it).
pub fn run(
    allocator: std.mem.Allocator,
    name: []const u8,
    options: RunOptions,
    comptime func: anytype,
    args: anytype,
) !RunResult {
    const inner: usize = if (options.inner_iters == 0) 1 else options.inner_iters;

    // Warmup.
    for (0..options.warmup_iters) |_| {
        @call(.auto, func, args);
    }

    var timer = std.time.Timer.start() catch return error.TimerUnsupported;

    const samples = try allocator.alloc(u64, options.samples);
    errdefer allocator.free(samples);

    for (samples) |*out| {
        timer.reset();
        for (0..inner) |_| {
            @call(.auto, func, args);
        }
        const total_ns = timer.read();
        out.* = total_ns / @as(u64, @intCast(inner));
    }

    return .{
        .name = name,
        .samples_ns = samples,
        .iters_per_sample = inner,
        .allocator = allocator,
    };
}

pub fn sortNs(samples_ns: []u64) void {
    std.sort.pdq(u64, samples_ns, {}, std.sort.asc(u64));
}

/// Percentile for *sorted ascending* samples using the "nearest rank" (index) rule.
pub fn percentileSortedNs(sorted_ns: []const u64, p: f64) u64 {
    std.debug.assert(sorted_ns.len > 0);
    std.debug.assert(p >= 0.0 and p <= 1.0);

    if (sorted_ns.len == 1) return sorted_ns[0];

    const n_minus_1: f64 = @floatFromInt(sorted_ns.len - 1);
    const idx_f = @floor(p * n_minus_1);
    const idx: usize = @intFromFloat(idx_f);
    return sorted_ns[@min(idx, sorted_ns.len - 1)];
}

pub fn medianSortedNs(sorted_ns: []const u64) u64 {
    return percentileSortedNs(sorted_ns, 0.5);
}

pub fn percentileNs(allocator: std.mem.Allocator, samples_ns: []const u64, p: f64) !u64 {
    const tmp = try allocator.dupe(u64, samples_ns);
    defer allocator.free(tmp);
    sortNs(tmp);
    return percentileSortedNs(tmp, p);
}

pub fn medianNs(allocator: std.mem.Allocator, samples_ns: []const u64) !u64 {
    return percentileNs(allocator, samples_ns, 0.5);
}

/// GFLOP/s for a given `flops` and `duration_ns`.
///
/// Note: because \(1\text{ GFLOP} = 10^9\) and \(1\text{ s} = 10^9\text{ ns}\),
/// we have `GFLOP/s = flops / duration_ns`.
pub fn gflops(flops: u64, duration_ns: u64) f64 {
    if (duration_ns == 0) return std.math.inf(f64);
    return @as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(duration_ns));
}

pub fn bytesPerSec(bytes: u64, duration_ns: u64) f64 {
    if (duration_ns == 0) return std.math.inf(f64);
    return @as(f64, @floatFromInt(bytes)) * std.time.ns_per_s / @as(f64, @floatFromInt(duration_ns));
}

pub fn gibPerSec(bytes: u64, duration_ns: u64) f64 {
    const gib: f64 = 1024.0 * 1024.0 * 1024.0;
    return bytesPerSec(bytes, duration_ns) / gib;
}


