const std = @import("std");

/// Load reference BLAS implementations (OpenBLAS / BLIS) at runtime to enable parity tests.
///
/// ## Configuration
/// - `BLAZT_ORACLE_LIB`: explicit path/name to the library to load (highest priority)
/// - `BLAZT_ORACLE_OPENBLAS`: path/name for OpenBLAS
/// - `BLAZT_ORACLE_BLIS`: path/name for BLIS
///
/// If no env vars are set, we try common sonames (e.g. `libopenblas.so`, `libblis.so`).
pub const Oracle = struct {
    lib: std.DynLib,
    kind: Kind,

    // We call GEMM via the Fortran BLAS entrypoints (`{s,d}gemm_` or `{s,d}gemm_64_`).
    // This avoids a Zig 0.16 indirect-call miscompile observed when calling CBLAS
    // symbols via function pointers loaded from `dlsym`.
    sgemm_f77: F77Sgemm,
    dgemm_f77: F77Dgemm,

    pub const Kind = enum { openblas, blis };

    pub const LoadError =
        error{ LibraryNotFound, SymbolNotFound } ||
        std.DynLib.Error ||
        std.process.GetEnvVarOwnedError;

    pub const CblasOrder = enum(c_int) {
        row_major = 101,
        col_major = 102,
    };

    pub const CblasTranspose = enum(c_int) {
        no_trans = 111,
        trans = 112,
        conj_trans = 113,
    };

    pub const BlasIntKind = enum { i32, i64 };

    pub const F77SgemmI32Fn = *const fn (
        transa: *const u8,
        transb: *const u8,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f32,
        a: [*]const f32,
        lda: *const c_int,
        b: [*]const f32,
        ldb: *const c_int,
        beta: *const f32,
        c: [*]f32,
        ldc: *const c_int,
    ) callconv(.c) void;

    pub const F77DgemmI32Fn = *const fn (
        transa: *const u8,
        transb: *const u8,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f64,
        a: [*]const f64,
        lda: *const c_int,
        b: [*]const f64,
        ldb: *const c_int,
        beta: *const f64,
        c: [*]f64,
        ldc: *const c_int,
    ) callconv(.c) void;

    pub const F77SgemmI64Fn = *const fn (
        transa: *const u8,
        transb: *const u8,
        m: *const c_long,
        n: *const c_long,
        k: *const c_long,
        alpha: *const f32,
        a: [*]const f32,
        lda: *const c_long,
        b: [*]const f32,
        ldb: *const c_long,
        beta: *const f32,
        c: [*]f32,
        ldc: *const c_long,
    ) callconv(.c) void;

    pub const F77DgemmI64Fn = *const fn (
        transa: *const u8,
        transb: *const u8,
        m: *const c_long,
        n: *const c_long,
        k: *const c_long,
        alpha: *const f64,
        a: [*]const f64,
        lda: *const c_long,
        b: [*]const f64,
        ldb: *const c_long,
        beta: *const f64,
        c: [*]f64,
        ldc: *const c_long,
    ) callconv(.c) void;

    pub const F77Sgemm = union(BlasIntKind) { i32: F77SgemmI32Fn, i64: F77SgemmI64Fn };
    pub const F77Dgemm = union(BlasIntKind) { i32: F77DgemmI32Fn, i64: F77DgemmI64Fn };

    pub fn unload(self: *Oracle) void {
        self.lib.close();
        self.* = undefined;
    }

    /// Load an oracle library from a path/name, but allow overriding required Fortran symbol names.
    /// This is primarily useful for tests that want to assert `error.SymbolNotFound`.
    pub fn loadFromPathWithFortranSymbols(
        path_or_name: []const u8,
        kind: Kind,
        sgemm_symbol: [:0]const u8, // e.g. "sgemm_" or "sgemm_64_"
        dgemm_symbol: [:0]const u8, // e.g. "dgemm_" or "dgemm_64_"
    ) LoadError!Oracle {
        var lib = try std.DynLib.open(path_or_name);
        errdefer lib.close();

        const f77 = try resolveFortran(&lib, sgemm_symbol, dgemm_symbol);
        return .{
            .lib = lib,
            .kind = kind,
            .sgemm_f77 = f77.sgemm_f77,
            .dgemm_f77 = f77.dgemm_f77,
        };
    }

    pub fn loadAny(allocator: std.mem.Allocator) LoadError!Oracle {
        if (try tryLoadKind(allocator, .openblas)) |o| return o;
        if (try tryLoadKind(allocator, .blis)) |o| return o;
        return error.LibraryNotFound;
    }

    pub fn loadOpenBLAS(allocator: std.mem.Allocator) LoadError!Oracle {
        return loadKind(allocator, .openblas);
    }

    pub fn loadBLIS(allocator: std.mem.Allocator) LoadError!Oracle {
        return loadKind(allocator, .blis);
    }

    pub fn loadKind(allocator: std.mem.Allocator, kind: Kind) LoadError!Oracle {
        // Highest priority: explicit override.
        if (try tryEnvPath(allocator, "BLAZT_ORACLE_LIB")) |p| {
            defer allocator.free(p);
            return loadFromPath(p, kind);
        }

        const kind_env = switch (kind) {
            .openblas => "BLAZT_ORACLE_OPENBLAS",
            .blis => "BLAZT_ORACLE_BLIS",
        };
        if (try tryEnvPath(allocator, kind_env)) |p| {
            defer allocator.free(p);
            return loadFromPath(p, kind);
        }

        // Fall back to common sonames.
        return loadFromCandidates(kindCandidates(kind), kind);
    }

    pub fn loadFromPath(path_or_name: []const u8, kind: Kind) LoadError!Oracle {
        var lib = try std.DynLib.open(path_or_name);
        errdefer lib.close();
        const f77 = resolveFortran(&lib, "sgemm_", "dgemm_") catch |err| switch (err) {
            error.SymbolNotFound => try resolveFortran(&lib, "sgemm_64_", "dgemm_64_"),
            else => return err,
        };

        return .{
            .lib = lib,
            .kind = kind,
            .sgemm_f77 = f77.sgemm_f77,
            .dgemm_f77 = f77.dgemm_f77,
        };
    }

    fn tryEnvPath(allocator: std.mem.Allocator, key: []const u8) LoadError!?[]u8 {
        return std.process.getEnvVarOwned(allocator, key) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => return err,
        };
    }

    fn tryLoadKind(allocator: std.mem.Allocator, kind: Kind) LoadError!?Oracle {
        return loadKind(allocator, kind) catch |err| switch (err) {
            error.LibraryNotFound => null,
            else => return err,
        };
    }

    fn loadFromCandidates(candidates: []const []const u8, kind: Kind) LoadError!Oracle {
        var first_non_notfound: ?std.DynLib.Error = null;

        for (candidates) |name| {
            const lib = std.DynLib.open(name) catch |err| {
                if (err == error.FileNotFound) {
                    continue;
                }
                if (first_non_notfound == null) first_non_notfound = err;
                continue;
            };
            var owned = lib;
            const f77 = resolveFortran(&owned, "sgemm_", "dgemm_") catch |err| switch (err) {
                error.SymbolNotFound => resolveFortran(&owned, "sgemm_64_", "dgemm_64_") catch |err2| switch (err2) {
                    error.SymbolNotFound => {
                        owned.close();
                        continue;
                    },
                    else => {
                        owned.close();
                        return err2;
                    },
                },
                else => {
                    owned.close();
                    return err;
                },
            };

            return .{
                .lib = owned,
                .kind = kind,
                .sgemm_f77 = f77.sgemm_f77,
                .dgemm_f77 = f77.dgemm_f77,
            };
        }

        if (first_non_notfound) |e| return e;
        return error.LibraryNotFound;
    }

    fn kindCandidates(kind: Kind) []const []const u8 {
        // Keep these conservative; users can always provide an explicit path.
        return switch (kind) {
            .openblas => &[_][]const u8{
                "libopenblas.so",
                "libopenblas.so.0",
                "libopenblas64_.so",
                "libopenblas64.so",
                // common local build locations (if user built deps manually)
                "deps/openblas/libopenblas.so",
                "deps/openblas/libopenblas.so.0",
                "deps/openblas/libopenblas64_.so",
                "deps/openblas/libopenblas64.so",
                // common system locations (especially for Fedora/RHEL)
                "/lib64/libopenblas.so",
                "/lib64/libopenblas.so.0",
                "/lib64/libopenblas64_.so",
                "/lib64/libopenblas64_.so.0",
                "/lib64/libopenblas64.so",
                "/lib64/libopenblas64.so.0",
                "/usr/lib64/libopenblas.so",
                "/usr/lib64/libopenblas.so.0",
                "/usr/lib64/libopenblas64_.so",
                "/usr/lib64/libopenblas64_.so.0",
                "/usr/lib64/libopenblas64.so",
                "/usr/lib64/libopenblas64.so.0",
            },
            .blis => &[_][]const u8{
                "libblis.so",
                "libblis.so.4",
                "libblis.so.3",
                "deps/blis/libblis.so",
                "deps/blis/libblis.so.4",
                "deps/blis/libblis.so.3",
                "/lib64/libblis.so",
                "/lib64/libblis.so.4",
                "/lib64/libblis.so.3",
                "/usr/lib64/libblis.so",
                "/usr/lib64/libblis.so.4",
                "/usr/lib64/libblis.so.3",
            },
        };
    }

    fn lookupSym(lib: *std.DynLib, name: [:0]const u8) ?*anyopaque {
        return lib.lookup(*anyopaque, name);
    }

    fn symToFn(comptime Fn: type, sym: *anyopaque) Fn {
        return @ptrFromInt(@intFromPtr(sym));
    }

    const FortranResolved = struct {
        sgemm_f77: F77Sgemm,
        dgemm_f77: F77Dgemm,
    };

    fn resolveFortran(
        lib: *std.DynLib,
        sgemm_symbol: [:0]const u8,
        dgemm_symbol: [:0]const u8,
    ) LoadError!FortranResolved {
        const sym_s = lookupSym(lib, sgemm_symbol) orelse return error.SymbolNotFound;
        const sym_d = lookupSym(lib, dgemm_symbol) orelse return error.SymbolNotFound;

        const is_64 = std.mem.endsWith(u8, sgemm_symbol, "64_") or std.mem.endsWith(u8, dgemm_symbol, "64_");

        return .{
            .sgemm_f77 = if (is_64)
                .{ .i64 = symToFn(F77SgemmI64Fn, sym_s) }
            else
                .{ .i32 = symToFn(F77SgemmI32Fn, sym_s) },
            .dgemm_f77 = if (is_64)
                .{ .i64 = symToFn(F77DgemmI64Fn, sym_d) }
            else
                .{ .i32 = symToFn(F77DgemmI32Fn, sym_d) },
        };
    }

    fn transChar(t: CblasTranspose) u8 {
        return switch (t) {
            .no_trans => 'N',
            .trans => 'T',
            .conj_trans => 'T', // real GEMM: treat conj-trans as trans
        };
    }

    /// CBLAS-style `sgemm`, implemented via Fortran `sgemm_`/`sgemm_64_`.
    /// Only `.col_major` is supported.
    pub fn sgemm(
        self: *const Oracle,
        order: CblasOrder,
        trans_a: CblasTranspose,
        trans_b: CblasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: [*]const f32,
        lda: c_int,
        b: [*]const f32,
        ldb: c_int,
        beta: f32,
        c: [*]f32,
        ldc: c_int,
    ) void {
        std.debug.assert(order == .col_major);

        const ta = transChar(trans_a);
        const tb = transChar(trans_b);

        var alpha_v = alpha;
        var beta_v = beta;

        switch (self.sgemm_f77) {
            .i32 => |fn_ptr| {
                var mm: c_int = m;
                var nn: c_int = n;
                var kk: c_int = k;
                var lda_v: c_int = lda;
                var ldb_v: c_int = ldb;
                var ldc_v: c_int = ldc;
                fn_ptr(&ta, &tb, &mm, &nn, &kk, &alpha_v, a, &lda_v, b, &ldb_v, &beta_v, c, &ldc_v);
            },
            .i64 => |fn_ptr| {
                var mm: c_long = @intCast(m);
                var nn: c_long = @intCast(n);
                var kk: c_long = @intCast(k);
                var lda_v: c_long = @intCast(lda);
                var ldb_v: c_long = @intCast(ldb);
                var ldc_v: c_long = @intCast(ldc);
                fn_ptr(&ta, &tb, &mm, &nn, &kk, &alpha_v, a, &lda_v, b, &ldb_v, &beta_v, c, &ldc_v);
            },
        }
    }

    /// CBLAS-style `dgemm`, implemented via Fortran `dgemm_`/`dgemm_64_`.
    /// Only `.col_major` is supported.
    pub fn dgemm(
        self: *const Oracle,
        order: CblasOrder,
        trans_a: CblasTranspose,
        trans_b: CblasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: [*]const f64,
        lda: c_int,
        b: [*]const f64,
        ldb: c_int,
        beta: f64,
        c: [*]f64,
        ldc: c_int,
    ) void {
        std.debug.assert(order == .col_major);

        const ta = transChar(trans_a);
        const tb = transChar(trans_b);

        var alpha_v = alpha;
        var beta_v = beta;

        switch (self.dgemm_f77) {
            .i32 => |fn_ptr| {
                var mm: c_int = m;
                var nn: c_int = n;
                var kk: c_int = k;
                var lda_v: c_int = lda;
                var ldb_v: c_int = ldb;
                var ldc_v: c_int = ldc;
                fn_ptr(&ta, &tb, &mm, &nn, &kk, &alpha_v, a, &lda_v, b, &ldb_v, &beta_v, c, &ldc_v);
            },
            .i64 => |fn_ptr| {
                var mm: c_long = @intCast(m);
                var nn: c_long = @intCast(n);
                var kk: c_long = @intCast(k);
                var lda_v: c_long = @intCast(lda);
                var ldb_v: c_long = @intCast(ldb);
                var ldc_v: c_long = @intCast(ldc);
                fn_ptr(&ta, &tb, &mm, &nn, &kk, &alpha_v, a, &lda_v, b, &ldb_v, &beta_v, c, &ldc_v);
            },
        }
    }
};


