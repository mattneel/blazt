const std = @import("std");
const blazt = @import("blazt");

test "gemm.computeTileParams returns comptime-known sane values (f32/f64)" {
    comptime {
        const p32 = blazt.gemm.computeTileParams(f32);
        const p64 = blazt.gemm.computeTileParams(f64);

        // comptime-known usage (array length)
        var _buf32: [p32.MR]u8 = undefined;
        var _buf64: [p64.NR]u8 = undefined;
        _ = .{ &_buf32, &_buf64 };

        // basic bounds
        std.debug.assert(p32.MR > 0 and p32.NR > 0 and p32.KC > 0 and p32.MC > 0 and p32.NC > 0);
        std.debug.assert(p64.MR > 0 and p64.NR > 0 and p64.KC > 0 and p64.MC > 0 and p64.NC > 0);

        // alignment/multiples
        std.debug.assert(p32.MC % p32.MR == 0);
        std.debug.assert(p32.NC % p32.NR == 0);
        std.debug.assert(p64.MC % p64.MR == 0);
        std.debug.assert(p64.NC % p64.NR == 0);

        // conservative upper bounds
        std.debug.assert(p32.MR <= 16);
        std.debug.assert(p64.MR <= 16);
        std.debug.assert(p32.NR <= 8);
        std.debug.assert(p64.NR <= 8);
        std.debug.assert(p32.KC <= 2048);
        std.debug.assert(p64.KC <= 2048);
        std.debug.assert(p32.NC <= 8192);
        std.debug.assert(p64.NC <= 8192);
    }
}


