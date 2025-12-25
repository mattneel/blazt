const std = @import("std");
const blazt = @import("blazt");

test "Matrix init sets shape and cache-line alignment (row-major)" {
    var m = try blazt.Matrix(f64, .row_major).init(std.testing.allocator, 3, 4);
    defer m.deinit();

    try std.testing.expectEqual(@as(usize, 3), m.rows);
    try std.testing.expectEqual(@as(usize, 4), m.cols);
    try std.testing.expectEqual(@as(usize, 4), m.stride);

    try std.testing.expect(@intFromPtr(m.data.ptr) % blazt.CacheLine == 0);
    try std.testing.expectEqual(@as(usize, 12), m.data.len);
}

test "Matrix indexing matches layout (row-major vs col-major)" {
    // row-major 2x3
    {
        var m = try blazt.Matrix(u32, .row_major).init(std.testing.allocator, 2, 3);
        defer m.deinit();

        for (m.data, 0..) |*v, idx| v.* = @intCast(idx);

        try std.testing.expectEqual(@as(u32, 0), m.at(0, 0));
        try std.testing.expectEqual(@as(u32, 1), m.at(0, 1));
        try std.testing.expectEqual(@as(u32, 2), m.at(0, 2));
        try std.testing.expectEqual(@as(u32, 3), m.at(1, 0));
        try std.testing.expectEqual(@as(u32, 4), m.at(1, 1));
        try std.testing.expectEqual(@as(u32, 5), m.at(1, 2));
    }

    // col-major 2x3
    {
        var m = try blazt.Matrix(u32, .col_major).init(std.testing.allocator, 2, 3);
        defer m.deinit();

        for (m.data, 0..) |*v, idx| v.* = @intCast(idx);

        try std.testing.expectEqual(@as(u32, 0), m.at(0, 0));
        try std.testing.expectEqual(@as(u32, 1), m.at(1, 0));
        try std.testing.expectEqual(@as(u32, 2), m.at(0, 1));
        try std.testing.expectEqual(@as(u32, 3), m.at(1, 1));
        try std.testing.expectEqual(@as(u32, 4), m.at(0, 2));
        try std.testing.expectEqual(@as(u32, 5), m.at(1, 2));
    }
}


