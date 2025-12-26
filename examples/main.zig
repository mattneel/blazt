const std = @import("std");
const blazt = @import("blazt");

fn exampleGemm(alloc: std.mem.Allocator) !void {
    var a = try blazt.Matrix(f64, .row_major).init(alloc, 2, 2);
    defer a.deinit();
    var b = try blazt.Matrix(f64, .row_major).init(alloc, 2, 2);
    defer b.deinit();
    var c = try blazt.Matrix(f64, .row_major).init(alloc, 2, 2);
    defer c.deinit();

    a.atPtr(0, 0).* = 1;
    a.atPtr(0, 1).* = 2;
    a.atPtr(1, 0).* = 3;
    a.atPtr(1, 1).* = 4;

    b.atPtr(0, 0).* = 5;
    b.atPtr(0, 1).* = 6;
    b.atPtr(1, 0).* = 7;
    b.atPtr(1, 1).* = 8;

    @memset(c.data, @as(f64, 0));

    blazt.ops.gemm(f64, .row_major, .no_trans, .no_trans, 1.0, a, b, 0.0, &c);

    std.debug.print(
        "example gemm (2x2)\n  C = [[{d:.3}, {d:.3}], [{d:.3}, {d:.3}]]\n",
        .{ c.at(0, 0), c.at(0, 1), c.at(1, 0), c.at(1, 1) },
    );
}

fn exampleLu(alloc: std.mem.Allocator) !void {
    var a = try blazt.Matrix(f64, .row_major).init(alloc, 3, 3);
    defer a.deinit();

    // A =
    // [ 2  1  1 ]
    // [ 4 -6  0 ]
    // [-2  7  2 ]
    a.atPtr(0, 0).* = 2;
    a.atPtr(0, 1).* = 1;
    a.atPtr(0, 2).* = 1;
    a.atPtr(1, 0).* = 4;
    a.atPtr(1, 1).* = -6;
    a.atPtr(1, 2).* = 0;
    a.atPtr(2, 0).* = -2;
    a.atPtr(2, 1).* = 7;
    a.atPtr(2, 2).* = 2;

    const ipiv = try alloc.alloc(i32, 3);
    defer alloc.free(ipiv);

    try blazt.ops.lu(f64, &a, ipiv);

    std.debug.print(
        "example lu (3x3)\n  ipiv = [{d}, {d}, {d}]\n",
        .{ ipiv[0], ipiv[1], ipiv[2] },
    );
}

fn exampleCholesky(alloc: std.mem.Allocator) !void {
    var a = try blazt.Matrix(f64, .row_major).init(alloc, 3, 3);
    defer a.deinit();

    // SPD matrix:
    // [ 4 12 -16]
    // [12 37 -43]
    // [-16 -43 98]
    a.atPtr(0, 0).* = 4;
    a.atPtr(0, 1).* = 12;
    a.atPtr(0, 2).* = -16;
    a.atPtr(1, 0).* = 12;
    a.atPtr(1, 1).* = 37;
    a.atPtr(1, 2).* = -43;
    a.atPtr(2, 0).* = -16;
    a.atPtr(2, 1).* = -43;
    a.atPtr(2, 2).* = 98;

    try blazt.ops.cholesky(f64, .lower, &a);

    std.debug.print(
        "example cholesky (3x3, lower)\n  L11={d:.3} L21={d:.3} L22={d:.3} L31={d:.3} L32={d:.3} L33={d:.3}\n",
        .{ a.at(0, 0), a.at(1, 0), a.at(1, 1), a.at(2, 0), a.at(2, 1), a.at(2, 2) },
    );
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    try exampleGemm(alloc);
    try exampleLu(alloc);
    try exampleCholesky(alloc);
}


