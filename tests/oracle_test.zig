const std = @import("std");
const blazt = @import("blazt");

test "oracle loader: OpenBLAS CBLAS symbols load and are callable" {
    var oracle = blazt.oracle.Oracle.loadOpenBLAS(std.testing.allocator) catch |err| switch (err) {
        error.LibraryNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer oracle.unload();

    // 1x1 GEMM sanity (for 1x1, row/col-major makes no difference).
    const a_d = [_]f64{1.0};
    const b_d = [_]f64{2.0};
    var c_d = [_]f64{3.0};
    oracle.dgemm(.col_major, .no_trans, .no_trans, 1, 1, 1, 1.0, a_d[0..].ptr, 1, b_d[0..].ptr, 1, 0.0, c_d[0..].ptr, 1);
    try std.testing.expectEqual(@as(f64, 2.0), c_d[0]);

    const a_s = [_]f32{1.0};
    const b_s = [_]f32{2.0};
    var c_s = [_]f32{3.0};
    oracle.sgemm(.col_major, .no_trans, .no_trans, 1, 1, 1, 1.0, a_s[0..].ptr, 1, b_s[0..].ptr, 1, 0.0, c_s[0..].ptr, 1);
    try std.testing.expectEqual(@as(f32, 2.0), c_s[0]);
}

test "oracle loader: SymbolNotFound is explicit for missing required symbols" {
    // Use OpenBLAS if present; we force a symbol miss to validate the error path.
    _ = blazt.oracle.Oracle.loadFromPathWithFortranSymbols(
        "libopenblas.so",
        .openblas,
        "sgemm_",
        "dgemm__blazt_missing_symbol",
    ) catch |err| switch (err) {
        error.FileNotFound, error.LibraryNotFound => return error.SkipZigTest,
        error.SymbolNotFound => return,
        else => return err,
    };
    // If it somehow loaded, that's a test failure.
    return error.TestUnexpectedError;
}
