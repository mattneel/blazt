//! FFI exports for freestanding RISC-V (Tensix) target.
//! These functions use C ABI and can be called from external code.

const blazt = @import("blazt");
// blazt.ops is already the ops struct (from ops.zig).ops
const ops = blazt.ops;
const Matrix = blazt.Matrix;
const Layout = blazt.Layout;
const Trans = blazt.Trans;
const CacheLine = blazt.CacheLine;

// ============================================================================
// BLAS Level 1: Vector operations
// ============================================================================

/// SAXPY: y = alpha * x + y (single precision)
export fn blazt_saxpy(n: usize, alpha: f32, x: [*]const f32, y: [*]f32) void {
    if (n == 0) return;
    ops.axpy(f32, n, alpha, x[0..n], y[0..n]);
}

/// DAXPY: y = alpha * x + y (double precision)
export fn blazt_daxpy(n: usize, alpha: f64, x: [*]const f64, y: [*]f64) void {
    if (n == 0) return;
    ops.axpy(f64, n, alpha, x[0..n], y[0..n]);
}

/// SDOT: dot product (single precision)
export fn blazt_sdot(n: usize, x: [*]const f32, y: [*]const f32) f32 {
    if (n == 0) return 0;
    return ops.dot(f32, x[0..n], y[0..n]);
}

/// DDOT: dot product (double precision)
export fn blazt_ddot(n: usize, x: [*]const f64, y: [*]const f64) f64 {
    if (n == 0) return 0;
    return ops.dot(f64, x[0..n], y[0..n]);
}

/// SSCAL: x = alpha * x (single precision)
export fn blazt_sscal(n: usize, alpha: f32, x: [*]f32) void {
    if (n == 0) return;
    ops.scal(f32, n, alpha, x[0..n]);
}

/// DSCAL: x = alpha * x (double precision)
export fn blazt_dscal(n: usize, alpha: f64, x: [*]f64) void {
    if (n == 0) return;
    ops.scal(f64, n, alpha, x[0..n]);
}

/// SNRM2: Euclidean norm (single precision)
export fn blazt_snrm2(n: usize, x: [*]const f32) f32 {
    if (n == 0) return 0;
    return ops.nrm2(f32, x[0..n]);
}

/// DNRM2: Euclidean norm (double precision)
export fn blazt_dnrm2(n: usize, x: [*]const f64) f64 {
    if (n == 0) return 0;
    return ops.nrm2(f64, x[0..n]);
}

// ============================================================================
// BLAS Level 3: Matrix-matrix operations
// ============================================================================

/// Helper to create a matrix view from raw pointer (no allocation)
fn matrixView(comptime T: type, comptime layout: Layout, rows: usize, cols: usize, ptr: [*]T) Matrix(T, layout) {
    const stride = switch (layout) {
        .row_major => cols,
        .col_major => rows,
    };
    const len = rows * cols;
    // Note: We're bypassing alignment check for freestanding.
    // Caller is responsible for proper alignment.
    return .{
        .data = @as([]align(CacheLine) T, @alignCast(ptr[0..len])),
        .rows = rows,
        .cols = cols,
        .stride = stride,
        .allocator = null,
    };
}

fn constMatrixView(comptime T: type, comptime layout: Layout, rows: usize, cols: usize, ptr: [*]const T) Matrix(T, layout) {
    const stride = switch (layout) {
        .row_major => cols,
        .col_major => rows,
    };
    const len = rows * cols;
    return .{
        .data = @as([]align(CacheLine) T, @alignCast(@constCast(ptr[0..len]))),
        .rows = rows,
        .cols = cols,
        .stride = stride,
        .allocator = null,
    };
}

/// SGEMM: C = alpha * A * B + beta * C (single precision, row-major)
/// m: rows of A and C
/// n: columns of B and C
/// k: columns of A and rows of B
export fn blazt_sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: [*]const f32,
    b: [*]const f32,
    beta: f32,
    c: [*]f32,
) void {
    if (m == 0 or n == 0 or k == 0) return;

    const a_mat = constMatrixView(f32, .row_major, m, k, a);
    const b_mat = constMatrixView(f32, .row_major, k, n, b);
    var c_mat = matrixView(f32, .row_major, m, n, c);

    ops.gemm(f32, .row_major, .no_trans, .no_trans, alpha, a_mat, b_mat, beta, &c_mat);
}

/// DGEMM: C = alpha * A * B + beta * C (double precision, row-major)
export fn blazt_dgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: [*]const f64,
    b: [*]const f64,
    beta: f64,
    c: [*]f64,
) void {
    if (m == 0 or n == 0 or k == 0) return;

    const a_mat = constMatrixView(f64, .row_major, m, k, a);
    const b_mat = constMatrixView(f64, .row_major, k, n, b);
    var c_mat = matrixView(f64, .row_major, m, n, c);

    ops.gemm(f64, .row_major, .no_trans, .no_trans, alpha, a_mat, b_mat, beta, &c_mat);
}

// ============================================================================
// BLAS Level 2: Matrix-vector operations
// ============================================================================

/// SGEMV: y = alpha * A * x + beta * y (single precision, row-major)
/// m: rows of A
/// n: columns of A (length of x)
export fn blazt_sgemv(
    m: usize,
    n: usize,
    alpha: f32,
    a: [*]const f32,
    x: [*]const f32,
    beta: f32,
    y: [*]f32,
) void {
    if (m == 0 or n == 0) return;

    const a_mat = constMatrixView(f32, .row_major, m, n, a);
    ops.gemv(f32, .row_major, .no_trans, m, n, alpha, a_mat, x[0..n], beta, y[0..m]);
}

/// DGEMV: y = alpha * A * x + beta * y (double precision, row-major)
export fn blazt_dgemv(
    m: usize,
    n: usize,
    alpha: f64,
    a: [*]const f64,
    x: [*]const f64,
    beta: f64,
    y: [*]f64,
) void {
    if (m == 0 or n == 0) return;

    const a_mat = constMatrixView(f64, .row_major, m, n, a);
    ops.gemv(f64, .row_major, .no_trans, m, n, alpha, a_mat, x[0..n], beta, y[0..m]);
}
