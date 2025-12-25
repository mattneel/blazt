const std = @import("std");
const blazt = @import("blazt");

test "sve.runtimeVlBytes is compile-safe on non-SVE targets" {
    // On non-aarch64+sve builds, this must not trap/compile-error and should return 0.
    if (!comptime blazt.sve.has_sve) {
        try std.testing.expectEqual(@as(usize, 0), blazt.sve.runtimeVlBytes());
    }
}

test "sve.dispatchFixedVectorLen selects expected candidate and falls back to scalar" {
    const Ctx = struct { got: usize = 0 };
    var ctx: Ctx = .{};

    const kernel = struct {
        fn f(comptime VL: usize, c: *Ctx) void {
            c.got = VL;
        }
    }.f;

    _ = blazt.sve.dispatchFixedVectorLen(&.{ 16, 8, 4 }, 10, kernel, &ctx);
    try std.testing.expectEqual(@as(usize, 8), ctx.got);

    ctx.got = 0;
    _ = blazt.sve.dispatchFixedVectorLen(&.{ 16, 8, 4 }, 2, kernel, &ctx);
    try std.testing.expectEqual(@as(usize, 1), ctx.got);
}


