const std = @import("std");
const blazt = @import("blazt");

test "core enums exist and can be referenced" {
    // Layout is defined on Matrix module and re-exported from root.
    try std.testing.expect(@typeInfo(blazt.Layout) == .@"enum");
    try std.testing.expect(@typeInfo(blazt.Trans) == .@"enum");
    try std.testing.expect(@typeInfo(blazt.UpLo) == .@"enum");
    try std.testing.expect(@typeInfo(blazt.Diag) == .@"enum");
}

test "error taxonomies compile and can be pattern-matched" {
    // This is a compile-time “shape check” that the error sets exist and are usable.
    const lu_err: blazt.LuError = error.Singular;
    const chol_err: blazt.CholeskyError = error.NotPositiveDefinite;
    const trsv_err: blazt.TrsvError = error.Singular;

    _ = switch (lu_err) {
        error.Singular => true,
    };
    _ = switch (chol_err) {
        error.NotPositiveDefinite => true,
    };
    _ = switch (trsv_err) {
        error.Singular => true,
    };
}


