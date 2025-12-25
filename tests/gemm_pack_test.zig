const std = @import("std");
const blazt = @import("blazt");

fn fillMatrix(comptime layout: blazt.Layout, m: usize, n: usize) !blazt.Matrix(f32, layout) {
    var a = try blazt.Matrix(f32, layout).init(std.testing.allocator, m, n);
    for (0..m) |i| {
        for (0..n) |j| {
            a.atPtr(i, j).* = @as(f32, @floatFromInt(@as(u32, @intCast(i * 100 + j))));
        }
    }
    return a;
}

test "gemm.packA packs MR×KC micro-panel with zero padding (row_major + col_major)" {
    const MR: usize = 4;
    const kc: usize = 3;
    const m_panel: usize = 3; // tail (m_panel < MR)
    const row0: usize = 1;
    const col0: usize = 1;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        var a = try fillMatrix(layout, 5, 4);
        defer a.deinit();

        const dst = try blazt.allocAligned(std.testing.allocator, f32, MR * kc);
        defer std.testing.allocator.free(dst);
        try std.testing.expect(@intFromPtr(dst.ptr) % blazt.CacheLine == 0);

        blazt.gemm.packA(f32, layout, MR, kc, m_panel, a, row0, col0, dst);

        // Validate packed layout: dst[p*MR+i] = A[row0+i, col0+p] for i<m_panel, else 0.
        for (0..kc) |p| {
            for (0..MR) |i| {
                const got = dst[p * MR + i];
                const expected: f32 = if (i < m_panel)
                    a.at(row0 + i, col0 + p)
                else
                    0;
                try std.testing.expectEqual(expected, got);
            }
        }
    }
}

test "gemm.packB packs KC×NR micro-panel with zero padding (row_major + col_major)" {
    const NR: usize = 4;
    const kc: usize = 3;
    const n_panel: usize = 2; // tail (n_panel < NR)
    const row0: usize = 2;
    const col0: usize = 1;

    inline for (.{ blazt.Layout.row_major, blazt.Layout.col_major }) |layout| {
        var b = try fillMatrix(layout, 6, 5);
        defer b.deinit();

        const dst = try blazt.allocAligned(std.testing.allocator, f32, NR * kc);
        defer std.testing.allocator.free(dst);
        try std.testing.expect(@intFromPtr(dst.ptr) % blazt.CacheLine == 0);

        blazt.gemm.packB(f32, layout, NR, kc, n_panel, b, row0, col0, dst);

        // Validate packed layout: dst[p*NR+j] = B[row0+p, col0+j] for j<n_panel, else 0.
        for (0..kc) |p| {
            for (0..NR) |j| {
                const got = dst[p * NR + j];
                const expected: f32 = if (j < n_panel)
                    b.at(row0 + p, col0 + j)
                else
                    0;
                try std.testing.expectEqual(expected, got);
            }
        }
    }
}


