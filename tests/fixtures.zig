const std = @import("std");
const blazt = @import("blazt");

pub const FixtureRng = struct {
    prng: std.Random.Xoshiro256,

    pub fn init(seed: u64) FixtureRng {
        return .{ .prng = std.Random.Xoshiro256.init(seed) };
    }

    pub fn random(self: *FixtureRng) std.Random {
        return self.prng.random();
    }
};

pub fn randomScalar(r: std.Random, comptime T: type) T {
    return switch (@typeInfo(T)) {
        .int => r.int(T),
        .float => blk: {
            // Spread around 0 to avoid pathological all-positive workloads.
            const x = r.float(T); // [0,1)
            break :blk (x * 2) - 1; // [-1, 1)
        },
        else => @compileError("randomScalar unsupported type: " ++ @typeName(T)),
    };
}

pub fn fillSliceRandom(r: std.Random, slice: anytype) void {
    const Slice = @typeInfo(@TypeOf(slice)).pointer;
    if (Slice.size != .slice) @compileError("fillSliceRandom expects a slice");
    const T = Slice.child;

    for (slice) |*x| {
        x.* = randomScalar(r, T);
    }
}

pub fn randomMatrix(
    r: std.Random,
    comptime T: type,
    comptime layout: blazt.Layout,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
) !blazt.Matrix(T, layout) {
    var m = try blazt.Matrix(T, layout).init(allocator, rows, cols);
    fillSliceRandom(r, m.data);
    return m;
}

pub const FloatTolerance = struct {
    abs: f64,
    rel: f64,
    ulps: u64,
};

pub fn defaultTolerance(comptime T: type) FloatTolerance {
    return switch (T) {
        f32 => .{ .abs = 1e-6, .rel = 1e-5, .ulps = 8 },
        f64 => .{ .abs = 1e-12, .rel = 1e-12, .ulps = 16 },
        else => @compileError("defaultTolerance only supports f32/f64"),
    };
}

fn isNan(x: anytype) bool {
    return switch (@typeInfo(@TypeOf(x))) {
        .float => std.math.isNan(x),
        else => false,
    };
}

fn isInf(x: anytype) bool {
    return switch (@typeInfo(@TypeOf(x))) {
        .float => std.math.isInf(x),
        else => false,
    };
}

fn floatOrderedBits(comptime T: type, x: T) std.meta.Int(.unsigned, @bitSizeOf(T)) {
    const U = std.meta.Int(.unsigned, @bitSizeOf(T));
    const bits: U = @bitCast(x);
    // Map IEEE-754 bits to monotonically increasing ordering:
    // flip sign bit for positives; invert all bits for negatives.
    const sign_mask: U = @as(U, 1) << (@bitSizeOf(T) - 1);
    return if ((bits & sign_mask) != 0) ~bits else bits ^ sign_mask;
}

pub fn ulpsDiff(comptime T: type, a: T, b: T) u64 {
    if (isNan(a) or isNan(b)) return std.math.maxInt(u64);
    const Ua = floatOrderedBits(T, a);
    const Ub = floatOrderedBits(T, b);
    const diff = if (Ua >= Ub) Ua - Ub else Ub - Ua;
    return @intCast(diff);
}

pub fn floatApproxEq(comptime T: type, a: T, b: T, tol: FloatTolerance) bool {
    // NaN policy: NaNs are never equal.
    if (isNan(a) or isNan(b)) return false;

    // Inf policy: must match exactly.
    if (isInf(a) or isInf(b)) return a == b;

    if (a == b) return true;

    const da: f64 = @floatCast(a);
    const db: f64 = @floatCast(b);
    const diff = @abs(da - db);

    const max_ab = @max(@abs(da), @abs(db));
    const ok_absrel = diff <= tol.abs + tol.rel * max_ab;
    if (ok_absrel) return true;

    return ulpsDiff(T, a, b) <= tol.ulps;
}

pub fn expectFloatApproxEq(comptime T: type, a: T, b: T, tol: FloatTolerance) !void {
    if (!floatApproxEq(T, a, b, tol)) {
        std.debug.print(
            "float mismatch:\n  a={d}\n  b={d}\n  abs_diff={d}\n  ulps={d}\n  tol.abs={d}\n  tol.rel={d}\n  tol.ulps={d}\n",
            .{ a, b, @abs(@as(f64, @floatCast(a)) - @as(f64, @floatCast(b))), ulpsDiff(T, a, b), tol.abs, tol.rel, tol.ulps },
        );
        return error.TestExpectedEqual;
    }
}

pub fn expectSliceApproxEq(comptime T: type, a: []const T, b: []const T, tol: FloatTolerance) !void {
    try std.testing.expectEqual(a.len, b.len);

    if (@typeInfo(T) != .float) {
        // For non-floats use strict equality for now.
        for (a, b, 0..) |av, bv, i| {
            if (av != bv) {
                std.debug.print("slice mismatch at {d}: a={any} b={any}\n", .{ i, av, bv });
                return error.TestExpectedEqual;
            }
        }
        return;
    }

    for (a, b, 0..) |av, bv, i| {
        if (!floatApproxEq(T, av, bv, tol)) {
            std.debug.print("slice mismatch at {d}\n", .{i});
            try expectFloatApproxEq(T, av, bv, tol);
        }
    }
}


