const std = @import("std");
const memory = @import("memory.zig");

pub const Layout = enum { row_major, col_major };

pub fn Matrix(comptime T: type, comptime layout: Layout) type {
    return struct {
        data: []align(memory.CacheLine) T,
        rows: usize,
        cols: usize,
        stride: usize,
        allocator: ?std.mem.Allocator,

        pub const Element = T;
        pub const LayoutType = Layout;
        pub const layout_const = layout;

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

        pub fn at(self: @This(), i: usize, j: usize) T {
            return self.data[self.index(i, j)];
        }

        pub fn atPtr(self: @This(), i: usize, j: usize) *T {
            return &self.data[self.index(i, j)];
        }
    };
}


