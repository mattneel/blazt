const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const cpu_cache = @import("cpu_cache");

extern fn blazt_nt_memset_zero(dst: [*]u8, n: usize) callconv(.c) void;

fn sanitizeCacheLineBytes(comptime line_bytes: usize) usize {
    const min_line: usize = std.atomic.cache_line;
    const base: usize = @max(line_bytes, min_line);
    if (base == 0) return 64;
    if (std.math.isPowerOfTwo(base)) return base;
    return std.math.ceilPowerOfTwo(usize, base) catch 64;
}

pub const CacheLine: usize = sanitizeCacheLineBytes(cpu_cache.l1d_line_bytes);

/// Zero a byte slice, optionally using non-temporal (streaming) stores when enabled.
pub fn memsetZeroBytes(bytes: []u8) void {
    if (bytes.len == 0) return;

    if (comptime build_options.nt_stores and builtin.cpu.arch == .x86_64) {
        if (bytes.len >= build_options.nt_store_min_bytes and (@intFromPtr(bytes.ptr) % 16) == 0) {
            blazt_nt_memset_zero(@ptrCast(bytes.ptr), bytes.len);
            return;
        }
    }

    @memset(bytes, 0);
}

pub fn allocAligned(
    allocator: std.mem.Allocator,
    comptime T: type,
    n: usize,
) ![]align(CacheLine) T {
    return try allocator.alignedAlloc(T, std.mem.Alignment.fromByteUnits(CacheLine), n);
}

fn PtrType(
    comptime size: std.builtin.Type.Pointer.Size,
    comptime alignment: usize,
    comptime is_const: bool,
    comptime is_volatile: bool,
    comptime Child: type,
) type {
    return switch (size) {
        .one => switch (@as(u2, @intFromBool(is_const)) << 1 | @as(u2, @intFromBool(is_volatile))) {
            0 => *align(alignment) Child,
            1 => *align(alignment) volatile Child,
            2 => *align(alignment) const Child,
            3 => *align(alignment) const volatile Child,
        },
        .many => switch (@as(u2, @intFromBool(is_const)) << 1 | @as(u2, @intFromBool(is_volatile))) {
            0 => [*]align(alignment) Child,
            1 => [*]align(alignment) volatile Child,
            2 => [*]align(alignment) const Child,
            3 => [*]align(alignment) const volatile Child,
        },
        .slice => switch (@as(u2, @intFromBool(is_const)) << 1 | @as(u2, @intFromBool(is_volatile))) {
            0 => []align(alignment) Child,
            1 => []align(alignment) volatile Child,
            2 => []align(alignment) const Child,
            3 => []align(alignment) const volatile Child,
        },
        .c => switch (@as(u2, @intFromBool(is_const)) << 1 | @as(u2, @intFromBool(is_volatile))) {
            0 => [*c]align(alignment) Child,
            1 => [*c]align(alignment) volatile Child,
            2 => [*c]align(alignment) const Child,
            3 => [*c]align(alignment) const volatile Child,
        },
    };
}

fn AlignedPointerType(comptime Ptr: type, comptime alignment: usize) type {
    const info = @typeInfo(Ptr);
    if (info != .pointer) {
        @compileError("ensureAligned expects a pointer type, got: " ++ @typeName(Ptr));
    }
    const p = info.pointer;

    if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) {
        @compileError("ensureAligned alignment must be a non-zero power of two");
    }
    if (p.address_space != .generic) {
        @compileError("ensureAligned does not support non-generic address spaces");
    }
    if (p.is_allowzero) {
        @compileError("ensureAligned does not support allowzero pointers");
    }
    if (p.sentinel_ptr != null) {
        @compileError("ensureAligned does not support sentinel pointers");
    }

    const current_alignment: usize = @intCast(p.alignment);
    const new_alignment = @max(current_alignment, alignment);
    return PtrType(p.size, new_alignment, p.is_const, p.is_volatile, p.child);
}

/// Runtime alignment assertion + aligned pointer cast.
///
/// This is intended for internal invariants and boundary checks (e.g. packed buffers).
pub fn ensureAligned(comptime alignment: usize, ptr: anytype) AlignedPointerType(@TypeOf(ptr), alignment) {
    const Ptr = @TypeOf(ptr);
    const info = @typeInfo(Ptr);
    if (info != .pointer) {
        @compileError("ensureAligned expects a pointer type, got: " ++ @typeName(Ptr));
    }

    if (info.pointer.size == .slice) {
        // Slices are { ptr, len }. Align the ptr and reconstruct the slice.
        if (@intFromPtr(ptr.ptr) % alignment != 0) {
            @panic("pointer not sufficiently aligned");
        }

        const current_alignment: usize = @intCast(info.pointer.alignment);
        const new_alignment = @max(current_alignment, alignment);
        const aligned_ptr_type = PtrType(.many, new_alignment, info.pointer.is_const, info.pointer.is_volatile, info.pointer.child);
        const aligned_ptr = @as(aligned_ptr_type, @alignCast(ptr.ptr));
        return aligned_ptr[0..ptr.len];
    }

    if (@intFromPtr(ptr) % alignment != 0) {
        @panic("pointer not sufficiently aligned");
    }

    return @as(AlignedPointerType(Ptr, alignment), @alignCast(ptr));
}
