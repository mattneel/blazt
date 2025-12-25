const std = @import("std");

pub const PrefetchOptions = std.builtin.PrefetchOptions;

/// Suggest a target-dependent vector length for `T`, or null if scalars are recommended.
pub fn suggestVectorLength(comptime T: type) ?comptime_int {
    return std.simd.suggestVectorLength(T);
}

/// Wrapper around the `@reduce` builtin.
pub inline fn reduce(comptime op: std.builtin.ReduceOp, vec: anytype) @TypeOf(@reduce(op, vec)) {
    return @reduce(op, vec);
}

pub inline fn mulAdd(a: anytype, b: anytype, c: anytype) @TypeOf(a) {
    // Zig 0.16: @mulAdd takes an explicit result type parameter.
    return @mulAdd(@TypeOf(a), a, b, c);
}

/// Wrapper around the `@prefetch` builtin.
pub inline fn prefetch(ptr: anytype, comptime options: PrefetchOptions) void {
    @prefetch(ptr, options);
}

fn vectorInfo(comptime V: type) std.builtin.Type.Vector {
    return switch (@typeInfo(V)) {
        .vector => |info| info,
        else => @compileError("expected vector type, got: " ++ @typeName(V)),
    };
}

/// Reverse a vector using `@shuffle`.
pub fn reverse(v: anytype) @TypeOf(v) {
    const V = @TypeOf(v);
    const info = vectorInfo(V);
    const T = info.child;
    const N = info.len;

    comptime var mask: [N]i32 = undefined;
    inline for (0..N) |i| {
        mask[i] = @intCast(N - 1 - i);
    }

    return @shuffle(T, v, undefined, mask);
}

/// Interleave lower halves: [a0, b0, a1, b1, ...] using the lower N/2 elements of a and b.
pub fn interleaveLower(a: anytype, b: anytype) @TypeOf(a) {
    const V = @TypeOf(a);
    const info = vectorInfo(V);
    const T = info.child;
    const N = info.len;

    if (@TypeOf(b) != V) @compileError("interleaveLower requires same vector type for a and b");
    if (N % 2 != 0) @compileError("interleaveLower requires even-length vectors");

    comptime var mask: [N]i32 = undefined;
    inline for (0..(N / 2)) |i| {
        const ii: i32 = @intCast(i);
        mask[2 * i] = ii;
        mask[2 * i + 1] = ~ii; // select from `b`
    }

    return @shuffle(T, a, b, mask);
}

/// Interleave upper halves: [a(N/2), b(N/2), ...] using the upper N/2 elements of a and b.
pub fn interleaveUpper(a: anytype, b: anytype) @TypeOf(a) {
    const V = @TypeOf(a);
    const info = vectorInfo(V);
    const T = info.child;
    const N = info.len;

    if (@TypeOf(b) != V) @compileError("interleaveUpper requires same vector type for a and b");
    if (N % 2 != 0) @compileError("interleaveUpper requires even-length vectors");

    comptime var mask: [N]i32 = undefined;
    inline for (0..(N / 2)) |i| {
        const idx = (N / 2) + i;
        const ii: i32 = @intCast(idx);
        mask[2 * i] = ii;
        mask[2 * i + 1] = ~ii; // select from `b`
    }

    return @shuffle(T, a, b, mask);
}


