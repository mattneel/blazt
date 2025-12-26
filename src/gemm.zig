const std = @import("std");
const cpu = @import("cpu.zig");
const simd = @import("simd.zig");
const memory = @import("memory.zig");
const matrix = @import("matrix.zig");
const build_options = @import("build_options");

pub const TileParams = struct {
    /// Micro-kernel rows.
    MR: usize,
    /// Micro-kernel cols.
    NR: usize,
    /// K blocking (fits A/B micro-panels in L1).
    KC: usize,
    /// M blocking (fits packed A in L2).
    MC: usize,
    /// N blocking (fits packed B in L3).
    NC: usize,
};

pub const KernelVariant = enum {
    generic,
    alpha_one,
    beta_zero,
    beta_one,
    alpha_one_beta_zero,
    alpha_one_beta_one,
};

pub const GemmTuningPreset = enum {
    /// Auto-select based on target CPU features.
    auto,
    /// Conservative baseline (cache-driven heuristics only).
    generic,
    /// x86 AVX2 (+FMA) tuned defaults.
    avx2,
    /// x86 AVX-512 tuned defaults.
    avx512,
    /// ARM NEON tuned defaults.
    neon,
};

fn parseGemmTuningPreset(comptime s: []const u8) GemmTuningPreset {
    if (std.mem.eql(u8, s, "auto")) return .auto;
    if (std.mem.eql(u8, s, "generic")) return .generic;
    if (std.mem.eql(u8, s, "avx2")) return .avx2;
    if (std.mem.eql(u8, s, "avx512")) return .avx512;
    if (std.mem.eql(u8, s, "neon")) return .neon;
    @compileError(std.fmt.comptimePrint("unknown -Dgemm_tuning preset '{s}' (expected auto|generic|avx2|avx512|neon)", .{s}));
}

fn autoGemmTuningPreset(info: cpu.CpuInfo) GemmTuningPreset {
    if (info.has_avx512f) return .avx512;
    if (info.has_avx2 and info.has_fma) return .avx2;
    if (info.has_neon) return .neon;
    return .generic;
}

fn selectedGemmTuningPreset(comptime info: cpu.CpuInfo, comptime requested: GemmTuningPreset) GemmTuningPreset {
    return switch (requested) {
        .auto => autoGemmTuningPreset(info),
        else => requested,
    };
}

fn autoPrefetchDistanceK(comptime T: type, comptime elems_per_k: usize) usize {
    // Heuristic: prefetch ~2 cache lines ahead, expressed in k-iterations.
    const bytes_per_k: usize = elems_per_k * @sizeOf(T);
    if (bytes_per_k == 0) return 0;
    const iters_per_line: usize = (memory.CacheLine + bytes_per_k - 1) / bytes_per_k;
    // Cap to avoid excessive lookahead on very small per-iter footprints.
    return @min(iters_per_line * 2, 64);
}

inline fn prefetchPackedPanels(
    comptime T: type,
    comptime MR: usize,
    comptime NR: usize,
    kc: usize,
    a: [*]const T,
    b: [*]const T,
    p: usize,
) void {
    if (!comptime build_options.gemm_prefetch) return;

    const opts: simd.PrefetchOptions = .{
        .rw = .read,
        .locality = build_options.gemm_prefetch_locality,
        .cache = .data,
    };

    const dist_a_k: usize = if (build_options.gemm_prefetch_a_k != 0)
        build_options.gemm_prefetch_a_k
    else
        autoPrefetchDistanceK(T, MR);
    const dist_b_k: usize = if (build_options.gemm_prefetch_b_k != 0)
        build_options.gemm_prefetch_b_k
    else
        autoPrefetchDistanceK(T, NR);

    if (dist_a_k != 0) {
        const p_a = p + dist_a_k;
        if (p_a < kc) simd.prefetch(&a[p_a * MR], opts);
    }
    if (dist_b_k != 0) {
        const p_b = p + dist_b_k;
        if (p_b < kc) simd.prefetch(&b[p_b * NR], opts);
    }
}

/// Choose a GEMM kernel specialization based on scalar values.
///
/// If called with comptime-known `alpha`/`beta`, the result is comptime-known as well.
pub fn selectGemmKernel(comptime T: type, alpha: T, beta: T) KernelVariant {
    if (comptime @typeInfo(T) != .float) {
        @compileError("selectGemmKernel only supports floating-point types");
    }

    const alpha_one = alpha == @as(T, 1);
    const beta_zero = beta == @as(T, 0);
    const beta_one = beta == @as(T, 1);

    if (alpha_one and beta_zero) return .alpha_one_beta_zero;
    if (alpha_one and beta_one) return .alpha_one_beta_one;
    if (alpha_one) return .alpha_one;
    if (beta_zero) return .beta_zero;
    if (beta_one) return .beta_one;
    return .generic;
}

fn roundDownMultiple(x: usize, multiple: usize) usize {
    if (multiple == 0) return x;
    return x - (x % multiple);
}

fn roundUpMultiple(x: usize, multiple: usize) usize {
    if (multiple == 0) return x;
    const r = x % multiple;
    return if (r == 0) x else x + (multiple - r);
}

fn clamp(x: usize, lo: usize, hi: usize) usize {
    return @min(@max(x, lo), hi);
}

/// Compute reasonable blocking parameters at comptime for GEMM kernels.
///
/// This is a **safe fallback** policy (not an auto-tuner):
/// - Derives MR from SIMD width
/// - Chooses NR conservatively
/// - Derives KC/MC/NC from conservative cache defaults (see `cpu.CacheDefaults`)
///
/// All returned values are **comptime-known** (no runtime dispatch).
pub fn computeTileParams(comptime T: type) TileParams {
    const requested = comptime parseGemmTuningPreset(build_options.gemm_tuning);
    return computeTileParamsWithTuning(T, requested);
}

/// Compute tile parameters using an explicit tuning preset.
///
/// This is mainly intended for testing and benchmarking presets; `computeTileParams(T)` uses
/// the build option `-Dgemm_tuning=...` (default: `auto`).
pub fn computeTileParamsWithTuning(comptime T: type, comptime tuning: GemmTuningPreset) TileParams {
    if (comptime @typeInfo(T) != .float) {
        @compileError("computeTileParams only supports floating-point types");
    }

    // These values are comptime-known for the target (features from `builtin.cpu`, caches from
    // build-time `cpu_cache`), but we must force comptime evaluation here so downstream selection
    // (tuning preset) is also comptime.
    const info = comptime cpu.CpuInfo.native();
    const vl_ci = simd.suggestVectorLength(T) orelse 1;
    const vl: usize = @intCast(vl_ci);

    const preset: GemmTuningPreset = comptime selectedGemmTuningPreset(info, tuning);

    // Micro-kernel register block sizes.
    // Keep these conservative so we don't over-commit registers on unknown targets.
    const MR: usize = clamp(vl, 1, 16);
    const NR: usize = switch (preset) {
        .generic => switch (T) {
            f32 => 4,
            f64 => 4,
            else => 2,
        },
        .avx2 => switch (T) {
            // BLIS-like defaults (initial pass).
            f32 => 6,
            f64 => 4,
            else => 2,
        },
        .avx512 => switch (T) {
            f32 => 8,
            f64 => 6,
            else => 2,
        },
        .neon => switch (T) {
            f32 => 4,
            f64 => 4,
            else => 2,
        },
        .auto => unreachable, // `selectedGemmTuningPreset` never returns .auto
    };

    const elt_bytes: usize = @sizeOf(T);
    const mrnr_bytes: usize = (MR + NR) * elt_bytes;

    // KC: keep A(mr,kc)+B(kc,nr) within ~half L1D (slightly more aggressive for known SIMD targets).
    const l1_budget = switch (preset) {
        .avx2, .avx512 => (info.l1d_size_bytes * 3) / 4,
        .neon, .generic => info.l1d_size_bytes / 2,
        .auto => unreachable,
    };
    const kc_raw = if (mrnr_bytes == 0) 64 else l1_budget / mrnr_bytes;
    const KC = clamp(roundDownMultiple(@max(kc_raw, 64), 8), 64, 2048);

    // MC: keep packed A(MC,KC) within ~half L2.
    const l2_budget = info.l2_size_bytes / 2;
    const mc_raw = if (KC == 0) MR else l2_budget / (KC * elt_bytes);
    const MC = blk: {
        const hi = roundDownMultiple(4096, MR);
        break :blk clamp(roundDownMultiple(@max(mc_raw, MR), MR), MR, hi);
    };

    // NC: keep packed B(KC,NC) within ~half L3.
    const l3_budget = info.l3_size_bytes / 2;
    const nc_raw = if (KC == 0) NR else l3_budget / (KC * elt_bytes);
    const NC = blk: {
        const hi = roundDownMultiple(8192, NR);
        break :blk clamp(roundDownMultiple(@max(nc_raw, NR), NR), NR, hi);
    };

    // If NC ended up smaller than a single cache line of B-panel, round it up slightly.
    const min_nc = roundUpMultiple(NR * 4, NR);
    const NC2 = if (NC < min_nc) min_nc else NC;

    return .{
        .MR = MR,
        .NR = NR,
        .KC = KC,
        .MC = MC,
        .NC = NC2,
    };
}

/// Number of B micro-panels to batch in the macro-kernel.
///
/// Batching reduces redundant A packing work: we pack one A micro-panel and reuse it across
/// multiple NR-wide micro-panels of B.
///
/// This must stay modest because pack buffers are stack-allocated in some call paths.
pub fn packBPanelCount(comptime T: type, P: TileParams) usize {
    const max_bytes: usize = 256 * 1024; // conservative per-call stack budget for packed B
    const stride_elems: usize = packBPanelStrideElems(T, P);
    const stride_bytes: usize = stride_elems * @sizeOf(T);
    if (stride_bytes == 0) return 1;
    const ratio: usize = max_bytes / stride_bytes;
    return clamp(ratio, 1, 8);
}

/// Stride (in elements) between consecutive packed B micro-panels within `pack_b`.
///
/// We round up to a cache-line multiple so each micro-panel starts cache-line aligned.
pub fn packBPanelStrideElems(comptime T: type, P: TileParams) usize {
    const align_elems: usize = memory.CacheLine / @sizeOf(T);
    return roundUpMultiple(P.NR * P.KC, align_elems);
}

/// Register-blocked GEMM micro-kernel for packed panels.
///
/// Computes an `MR x NR` block:
/// \[
///   C := \alpha (A_{MR\times KC} \cdot B_{KC\times NR}) + \beta C
/// \]
///
/// - `a` is packed as **KC blocks of MR contiguous elements** (A panel)
/// - `b` is packed as **KC blocks of NR contiguous elements** (B panel)
/// - `c` is addressed using row/col strides: `c[i*rs_c + j*cs_c]`
/// - Uses `@mulAdd` (FMA) on vectors for the accumulation loop.
///
/// Note: This kernel relies on floating-point fused multiply-add semantics when available; results
/// are validated against a naive reference in tests (within tolerance).
pub fn microKernel(
    comptime T: type,
    comptime MR: usize,
    comptime NR: usize,
    kc: usize,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    rs_c: usize,
    cs_c: usize,
    alpha: T,
    beta: T,
) void {
    if (comptime @typeInfo(T) != .float) {
        @compileError("microKernel only supports floating-point types");
    }
    if (MR == 0 or NR == 0) return;

    const Vec = @Vector(MR, T);

    // Accumulators: one vector per output column.
    var acc: [NR]Vec = undefined;
    inline for (0..NR) |j| {
        acc[j] = @splat(@as(T, 0));
    }

    var p: usize = 0;
    while (p < kc) : (p += 1) {
        prefetchPackedPanels(T, MR, NR, kc, a, b, p);
        const va: Vec = @as(*align(1) const Vec, @ptrCast(a + p * MR)).*;

        inline for (0..NR) |j| {
            const vb: Vec = @splat(b[p * NR + j]);
            acc[j] = simd.mulAdd(va, vb, acc[j]);
        }
    }

    // Store/update C.
    if (rs_c == 1) {
        const alpha_v: Vec = @splat(alpha);
        if (beta == @as(T, 0)) {
            // beta=0 fast path: do not read C.
            inline for (0..NR) |j| {
                const out: Vec = alpha_v * acc[j];
                const ptr_t: [*]T = c + j * cs_c;
                const ptr_v: *align(1) Vec = @ptrCast(ptr_t);
                ptr_v.* = out;
            }
        } else if (beta == @as(T, 1)) {
            inline for (0..NR) |j| {
                const ptr_t: [*]T = c + j * cs_c;
                const ptr_v: *align(1) Vec = @ptrCast(ptr_t);
                const vc: Vec = ptr_v.*;
                const out: Vec = alpha_v * acc[j] + vc;
                ptr_v.* = out;
            }
        } else {
            const beta_v: Vec = @splat(beta);
            inline for (0..NR) |j| {
                const ptr_t: [*]T = c + j * cs_c;
                const ptr_v: *align(1) Vec = @ptrCast(ptr_t);
                const vc: Vec = ptr_v.*;
                const out: Vec = alpha_v * acc[j] + beta_v * vc;
                ptr_v.* = out;
            }
        }
    } else {
        // Generic strided path: scalar updates.
        if (beta == @as(T, 0)) {
            // beta=0 fast path: do not read C.
            inline for (0..NR) |j| {
                const col = acc[j];
                inline for (0..MR) |i| {
                    c[i * rs_c + j * cs_c] = alpha * col[i];
                }
            }
        } else if (beta == @as(T, 1)) {
            inline for (0..NR) |j| {
                const col = acc[j];
                inline for (0..MR) |i| {
                    const idx = i * rs_c + j * cs_c;
                    c[idx] = alpha * col[i] + c[idx];
                }
            }
        } else {
            inline for (0..NR) |j| {
                const col = acc[j];
                inline for (0..MR) |i| {
                    const idx = i * rs_c + j * cs_c;
                    c[idx] = alpha * col[i] + beta * c[idx];
                }
            }
        }
    }
}

/// Tail-safe variant of `microKernel`: writes only an `mr x nr` sub-block (mr<=MR, nr<=NR).
///
/// This is used by the macro-kernel to handle edge tiles without requiring a temporary C buffer.
pub fn microKernelPartial(
    comptime T: type,
    comptime MR: usize,
    comptime NR: usize,
    kc: usize,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    rs_c: usize,
    cs_c: usize,
    alpha: T,
    beta: T,
    mr: usize,
    nr: usize,
) void {
    if (MR == 0 or NR == 0) return;
    const mr_store: usize = @min(mr, MR);
    const nr_store: usize = @min(nr, NR);

    // Compute full MR×NR accumulators (packed panels are zero-padded for tails).
    const Vec = @Vector(MR, T);
    var acc: [NR]Vec = undefined;
    inline for (0..NR) |j| acc[j] = @splat(@as(T, 0));

    var p: usize = 0;
    while (p < kc) : (p += 1) {
        prefetchPackedPanels(T, MR, NR, kc, a, b, p);
        const va: Vec = @as(*align(1) const Vec, @ptrCast(a + p * MR)).*;
        inline for (0..NR) |j| {
            const vb: Vec = @splat(b[p * NR + j]);
            acc[j] = simd.mulAdd(va, vb, acc[j]);
        }
    }

    // Store/update only the valid mr×nr region.
    if (beta == @as(T, 0)) {
        for (0..nr_store) |j| {
            const col_arr: [MR]T = @bitCast(acc[j]);
            for (0..mr_store) |i| {
                c[i * rs_c + j * cs_c] = alpha * col_arr[i];
            }
        }
    } else if (beta == @as(T, 1)) {
        for (0..nr_store) |j| {
            const col_arr: [MR]T = @bitCast(acc[j]);
            for (0..mr_store) |i| {
                const idx = i * rs_c + j * cs_c;
                c[idx] = alpha * col_arr[i] + c[idx];
            }
        }
    } else {
        for (0..nr_store) |j| {
            const col_arr: [MR]T = @bitCast(acc[j]);
            for (0..mr_store) |i| {
                const idx = i * rs_c + j * cs_c;
                c[idx] = alpha * col_arr[i] + beta * c[idx];
            }
        }
    }
}

/// Cache-blocked GEMM macro-kernel (col-major, no-transpose) using packing + `microKernel`.
///
/// Computes `C := alpha*A*B + beta*C` for:
/// - `A` (m×k), `B` (k×n), `C` (m×n)
/// - all matrices stored column-major (`layout = .col_major`)
///
/// **No hidden allocations**: callers must provide cache-line aligned pack buffers:
/// - `pack_a` length >= `P.MR * P.KC`
/// - `pack_b` length >= `packBPanelStrideElems(T, P) * packBPanelCount(T, P)`
pub fn gemmBlocked(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: matrix.Matrix(T, .col_major),
    b: matrix.Matrix(T, .col_major),
    beta: T,
    c: *matrix.Matrix(T, .col_major),
    pack_a: []align(memory.CacheLine) T,
    pack_b: []align(memory.CacheLine) T,
) void {
    gemmBlockedRange(T, m, n, k, alpha, a, b, beta, c, pack_a, pack_b, 0, n);
}

/// Cache-blocked GEMM macro-kernel restricted to a column range of C/B.
///
/// Like `gemmBlocked`, but computes only columns `jc0..jc1` (0 <= jc0 <= jc1 <= n).
pub fn gemmBlockedRange(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: matrix.Matrix(T, .col_major),
    b: matrix.Matrix(T, .col_major),
    beta: T,
    c: *matrix.Matrix(T, .col_major),
    pack_a: []align(memory.CacheLine) T,
    pack_b: []align(memory.CacheLine) T,
    jc0: usize,
    jc1: usize,
) void {
    if (comptime @typeInfo(T) != .float) {
        @compileError("gemmBlocked only supports floating-point types");
    }

    std.debug.assert(m == a.rows);
    std.debug.assert(k == a.cols);
    std.debug.assert(k == b.rows);
    std.debug.assert(n == b.cols);
    std.debug.assert(m == c.rows);
    std.debug.assert(n == c.cols);

    std.debug.assert(jc0 <= jc1);
    std.debug.assert(jc1 <= n);

    if (m == 0 or n == 0 or jc0 == jc1) return;

    // alpha==0 or k==0 => C := beta*C
    if (k == 0 or alpha == @as(T, 0)) {
        if (beta == @as(T, 0)) {
            // Zero only the selected column range.
            for (jc0..jc1) |j| {
                const col_off = j * c.stride;
                memory.memsetZeroBytes(std.mem.sliceAsBytes(c.data[col_off .. col_off + m]));
            }
        } else if (beta != @as(T, 1)) {
            // Scale only the selected column range.
            for (jc0..jc1) |j| {
                const col_off = j * c.stride;
                for (0..m) |i| c.data[col_off + i] *= beta;
            }
        }
        return;
    }

    const P: TileParams = comptime computeTileParams(T);
    const MR: usize = P.MR;
    const NR: usize = P.NR;
    const KC: usize = P.KC;
    const MC: usize = P.MC;
    const NC: usize = P.NC;

    std.debug.assert(pack_a.len >= MR * KC);
    const PB_STRIDE_ELEMS: usize = comptime packBPanelStrideElems(T, P);
    const PB_PANELS: usize = comptime packBPanelCount(T, P);
    std.debug.assert(pack_b.len >= PB_STRIDE_ELEMS * PB_PANELS);

    const rs_c: usize = 1;
    const cs_c: usize = c.stride;

    const use_pack_a_block: bool = pack_a.len >= MC * KC;

    var jc: usize = jc0;
    while (jc < jc1) : (jc += NC) {
        const nc_cur = @min(NC, jc1 - jc);

        var pc: usize = 0;
        while (pc < k) : (pc += KC) {
            const kc_cur = @min(KC, k - pc);
            const beta_block: T = if (pc == 0) beta else @as(T, 1);

            if (use_pack_a_block) {
                // Pack the entire MC×KC A block once (into pack_a) and reuse it across all B panels.
                //
                // This reduces redundant A packing work significantly for large N.
                var ic: usize = 0;
                while (ic < m) : (ic += MC) {
                    const mc_cur = @min(MC, m - ic);
                    const mc_pad = roundUpMultiple(mc_cur, MR);

                    // Packed A block size: mc_pad × kc_cur, arranged as MR-sized micro-panels.
                    // Indexing: for micro-panel starting at `ir`, element at (p,i) is:
                    //   pack_a_block[ir*kc_cur + p*MR + i]
                    const pack_a_block: []align(memory.CacheLine) T = pack_a[0 .. mc_pad * kc_cur];

                    // Pack A block (col_major) into pack_a_block.
                    var ir_pack: usize = 0;
                    while (ir_pack < mc_pad) : (ir_pack += MR) {
                        const row_base = ic + ir_pack;
                        for (0..kc_cur) |p| {
                            const src_col_off = (pc + p) * a.stride;
                            const dst_base = ir_pack * kc_cur + p * MR;
                            inline for (0..MR) |i| {
                                const r = row_base + i;
                                pack_a_block[dst_base + i] = if (r < ic + mc_cur) a.data[src_col_off + r] else @as(T, 0);
                            }
                        }
                    }

                    // Iterate NR-wide panels of B (in small batches) and compute.
                    const nr_block_cols: usize = NR * PB_PANELS;
                    var jr0: usize = 0;
                    while (jr0 < nc_cur) : (jr0 += nr_block_cols) {
                        const rem_cols: usize = nc_cur - jr0;
                        const jb_cur: usize = @min(PB_PANELS, (rem_cols + NR - 1) / NR);

                        // Pack JB micro-panels of B into pack_b.
                        var bj: usize = 0;
                        while (bj < jb_cur) : (bj += 1) {
                            const jr: usize = jr0 + bj * NR;
                            const nr_cur = @min(NR, nc_cur - jr);

                            const pb_ptr_unaligned: [*]T = pack_b.ptr + bj * PB_STRIDE_ELEMS;
                            const pb_ptr: [*]align(memory.CacheLine) T = @ptrCast(@alignCast(pb_ptr_unaligned));
                            const pb: []align(memory.CacheLine) T = pb_ptr[0 .. NR * kc_cur];
                            packB(T, .col_major, NR, kc_cur, nr_cur, b, pc, jc + jr, pb);
                        }

                        var ir: usize = 0;
                        while (ir < mc_cur) : (ir += MR) {
                            const mr_cur = @min(MR, mc_cur - ir);
                            const pa_ptr: [*]const T = pack_a_block.ptr + ir * kc_cur;

                            bj = 0;
                            while (bj < jb_cur) : (bj += 1) {
                                const jr: usize = jr0 + bj * NR;
                                const nr_cur = @min(NR, nc_cur - jr);

                                const pb_ptr_unaligned: [*]const T = pack_b.ptr + bj * PB_STRIDE_ELEMS;
                                const pb_ptr: [*]align(memory.CacheLine) const T = @ptrCast(@alignCast(pb_ptr_unaligned));
                                const c_ptr: [*]T = c.data[(jc + jr) * c.stride + (ic + ir) ..].ptr;
                                microKernelPartial(T, MR, NR, kc_cur, pa_ptr, pb_ptr, c_ptr, rs_c, cs_c, alpha, beta_block, mr_cur, nr_cur);
                            }
                        }
                    }
                }
            } else {
                // Fallback: A micro-panel packing reused across a small batch of B micro-panels.
                const nr_block_cols: usize = NR * PB_PANELS;
                var jr0: usize = 0;
                while (jr0 < nc_cur) : (jr0 += nr_block_cols) {
                    const rem_cols: usize = nc_cur - jr0;
                    const jb_cur: usize = @min(PB_PANELS, (rem_cols + NR - 1) / NR);

                    // Pack JB micro-panels of B into pack_b (each panel starts cache-line aligned).
                    var bj: usize = 0;
                    while (bj < jb_cur) : (bj += 1) {
                        const jr: usize = jr0 + bj * NR;
                        const nr_cur = @min(NR, nc_cur - jr);

                        const pb_ptr_unaligned: [*]T = pack_b.ptr + bj * PB_STRIDE_ELEMS;
                        const pb_ptr: [*]align(memory.CacheLine) T = @ptrCast(@alignCast(pb_ptr_unaligned));
                        const pb: []align(memory.CacheLine) T = pb_ptr[0 .. NR * kc_cur];
                        packB(T, .col_major, NR, kc_cur, nr_cur, b, pc, jc + jr, pb);
                    }

                    var ic: usize = 0;
                    while (ic < m) : (ic += MC) {
                        const mc_cur = @min(MC, m - ic);

                        var ir: usize = 0;
                        while (ir < mc_cur) : (ir += MR) {
                            const mr_cur = @min(MR, mc_cur - ir);

                            // Pack one MR×KC micro-panel of A (reused across JB B panels).
                            const pa = pack_a[0 .. MR * kc_cur];
                            packA(T, .col_major, MR, kc_cur, mr_cur, a, ic + ir, pc, pa);

                            bj = 0;
                            while (bj < jb_cur) : (bj += 1) {
                                const jr: usize = jr0 + bj * NR;
                                const nr_cur = @min(NR, nc_cur - jr);

                                const pb_ptr_unaligned: [*]const T = pack_b.ptr + bj * PB_STRIDE_ELEMS;
                                const pb_ptr: [*]align(memory.CacheLine) const T = @ptrCast(@alignCast(pb_ptr_unaligned));
                                const c_ptr: [*]T = c.data[(jc + jr) * c.stride + (ic + ir) ..].ptr;
                                microKernelPartial(T, MR, NR, kc_cur, pa.ptr, pb_ptr, c_ptr, rs_c, cs_c, alpha, beta_block, mr_cur, nr_cur);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Pack an `m x kc` micro-panel of `A` into contiguous storage for `microKernel`.
///
/// Packed layout (column panel):
/// - For each `p in 0..kc`, stores `MR` elements: `dst[p*MR + i] = A[row0 + i, col0 + p]`
/// - If `m < MR`, the remaining rows are **zero-padded**.
///
/// **Alignment**: `dst` must be cache-line aligned (`memory.CacheLine`).
pub fn packA(
    comptime T: type,
    comptime layout: matrix.Layout,
    comptime MR: usize,
    kc: usize,
    m: usize,
    a: matrix.Matrix(T, layout),
    row0: usize,
    col0: usize,
    dst: []align(memory.CacheLine) T,
) void {
    std.debug.assert(MR > 0);
    std.debug.assert(m <= MR);
    std.debug.assert(row0 + m <= a.rows);
    std.debug.assert(col0 + kc <= a.cols);
    std.debug.assert(dst.len >= MR * kc);
    std.debug.assert(@intFromPtr(dst.ptr) % memory.CacheLine == 0);

    const aAt = struct {
        inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    for (0..kc) |p| {
        inline for (0..MR) |i| {
            dst[p * MR + i] = if (i < m) aAt(a, row0 + i, col0 + p) else @as(T, 0);
        }
    }
}

/// Pack a `kc x n` micro-panel of `B` into contiguous storage for `microKernel`.
///
/// Packed layout (row panel):
/// - For each `p in 0..kc`, stores `NR` elements: `dst[p*NR + j] = B[row0 + p, col0 + j]`
/// - If `n < NR`, the remaining cols are **zero-padded**.
///
/// **Alignment**: `dst` must be cache-line aligned (`memory.CacheLine`).
pub fn packB(
    comptime T: type,
    comptime layout: matrix.Layout,
    comptime NR: usize,
    kc: usize,
    n: usize,
    b: matrix.Matrix(T, layout),
    row0: usize,
    col0: usize,
    dst: []align(memory.CacheLine) T,
) void {
    std.debug.assert(NR > 0);
    std.debug.assert(n <= NR);
    std.debug.assert(row0 + kc <= b.rows);
    std.debug.assert(col0 + n <= b.cols);
    std.debug.assert(dst.len >= NR * kc);
    std.debug.assert(@intFromPtr(dst.ptr) % memory.CacheLine == 0);

    const bAt = struct {
        inline fn f(mat: matrix.Matrix(T, layout), i: usize, j: usize) T {
            return switch (layout) {
                .row_major => mat.data[i * mat.stride + j],
                .col_major => mat.data[j * mat.stride + i],
            };
        }
    }.f;

    for (0..kc) |p| {
        inline for (0..NR) |j| {
            dst[p * NR + j] = if (j < n) bAt(b, row0 + p, col0 + j) else @as(T, 0);
        }
    }
}
