//! Basic matrix container and layout utilities.
//!
//! `blazt` matrices are simple, allocator-aware, cache-line aligned, contiguous buffers plus
//! shape/stride metadata. Most routines operate on `Matrix(T, layout)` values directly.

const std = @import("std");
const memory = @import("memory.zig");

/// Storage order for matrices.
///
/// - `.row_major`: elements are contiguous across columns (`stride = cols`).
/// - `.col_major`: elements are contiguous across rows (`stride = rows`).
pub const Layout = enum { row_major, col_major };

/// A dense matrix with a fixed compile-time element type and layout.
///
/// - **Data** is cache-line aligned (see `blazt.CacheLine`).
/// - **Stride** is expressed in elements (not bytes).
/// - `init()` allocates; `deinit()` frees only if `allocator != null` (so views can be represented).
pub fn Matrix(comptime T: type, comptime layout: Layout) type {
    return struct {
        /// Backing storage (contiguous).
        data: []align(memory.CacheLine) T,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
        /// Leading dimension (elements): `.row_major => cols`, `.col_major => rows`.
        stride: usize,
        /// Allocator used for `init()`. `null` means "do not free" (for views).
        allocator: ?std.mem.Allocator,

        pub const Element = T;
        pub const LayoutType = Layout;
        pub const layout_const = layout;

        /// Allocate a new dense matrix.
        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !@This() {
            const stride = switch (layout) {
                .row_major => cols,
                .col_major => rows,
            };

            const count = std.math.mul(usize, rows, cols) catch return error.OutOfMemory;
            const data = try memory.allocAligned(allocator, T, count);

            return .{
                .data = data,
                .rows = rows,
                .cols = cols,
                .stride = stride,
                .allocator = allocator,
            };
        }

        /// Free the backing storage if this matrix owns it.
        pub fn deinit(self: *@This()) void {
            const allocator = self.allocator orelse return;
            allocator.free(self.data);
            self.* = undefined;
        }

        fn index(self: @This(), i: usize, j: usize) usize {
            std.debug.assert(i < self.rows);
            std.debug.assert(j < self.cols);

            return switch (layout) {
                .row_major => i * self.stride + j,
                .col_major => j * self.stride + i,
            };
        }

        /// Read an element at `(i,j)`.
        pub fn at(self: @This(), i: usize, j: usize) T {
            return self.data[self.index(i, j)];
        }

        /// Get a pointer to an element at `(i,j)`.
        pub fn atPtr(self: @This(), i: usize, j: usize) *T {
            return &self.data[self.index(i, j)];
        }
    };
}


