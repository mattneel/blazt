const std = @import("std");
const blazt = @import("blazt");
const fx = @import("fixtures.zig");

test "FixtureRng is deterministic for a given seed" {
    var r1 = fx.FixtureRng.init(0x1234_5678_9abc_def0);
    var r2 = fx.FixtureRng.init(0x1234_5678_9abc_def0);

    const a1 = r1.random().int(u64);
    const a2 = r2.random().int(u64);
    try std.testing.expectEqual(a1, a2);

    const b1 = r1.random().int(u64);
    const b2 = r2.random().int(u64);
    try std.testing.expectEqual(b1, b2);
}

test "randomMatrix produces correct shape and non-trivial data" {
    var rng = fx.FixtureRng.init(42);
    var m = try fx.randomMatrix(rng.random(), f64, .row_major, std.testing.allocator, 4, 3);
    defer m.deinit();

    try std.testing.expectEqual(@as(usize, 4), m.rows);
    try std.testing.expectEqual(@as(usize, 3), m.cols);
    try std.testing.expect(@intFromPtr(m.data.ptr) % blazt.CacheLine == 0);

    var any_nonzero = false;
    for (m.data) |x| {
        if (x != 0) {
            any_nonzero = true;
            break;
        }
    }
    try std.testing.expect(any_nonzero);
}

test "floatApproxEq respects abs/rel/ulps tolerance" {
    const tol_f64 = fx.defaultTolerance(f64);

    // Exact equality.
    try std.testing.expect(fx.floatApproxEq(f64, 1.0, 1.0, tol_f64));

    // Small abs diff.
    try std.testing.expect(fx.floatApproxEq(f64, 1.0, 1.0 + 1e-13, tol_f64));

    // Big diff should fail.
    try std.testing.expect(!fx.floatApproxEq(f64, 1.0, 1.1, tol_f64));

    // Inf/NaN handling.
    try std.testing.expect(fx.floatApproxEq(f64, std.math.inf(f64), std.math.inf(f64), tol_f64));
    try std.testing.expect(!fx.floatApproxEq(f64, std.math.inf(f64), -std.math.inf(f64), tol_f64));
    try std.testing.expect(!fx.floatApproxEq(f64, std.math.nan(f64), std.math.nan(f64), tol_f64));
}


