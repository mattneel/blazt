const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

fn nrm2RefF64(x: []const f32) f32 {
    var acc: f64 = 0.0;
    for (x) |v| {
        const vv: f64 = @floatCast(v);
        acc += vv * vv;
    }
    return @floatCast(std.math.sqrt(acc));
}

fn asumRefF64(x: []const f32) f32 {
    var acc: f64 = 0.0;
    for (x) |v| acc += @abs(@as(f64, @floatCast(v)));
    return @floatCast(acc);
}

test "ops.nrm2 zero vector is zero" {
    var x: [128]f32 = .{0} ** 128;
    const got = blazt.ops.nrm2(f32, &x);
    try std.testing.expectEqual(@as(f32, 0.0), got);
}

test "ops.nrm2 handles large magnitudes without overflow" {
    const x: [2]f32 = .{ 1e20, 1e20 };
    const got = blazt.ops.nrm2(f32, &x);
    const ref = nrm2RefF64(x[0..]);
    try fx.expectFloatApproxEq(f32, ref, got, fx.defaultTolerance(f32));
    try std.testing.expect(!std.math.isInf(got));
    try std.testing.expect(!std.math.isNan(got));
}

test "ops.asum zero vector is zero" {
    var x: [256]f32 = .{0} ** 256;
    const got = blazt.ops.asum(f32, &x);
    try std.testing.expectEqual(@as(f32, 0.0), got);
}

test "ops.asum handles large magnitudes" {
    const x: [2]f32 = .{ 1e20, -1e20 };
    const got = blazt.ops.asum(f32, &x);
    const ref = asumRefF64(x[0..]);
    try fx.expectFloatApproxEq(f32, ref, got, fx.defaultTolerance(f32));
}


