//! Test harness for running blazt on Whisper RISC-V ISS.
//! Compile: zig build --build-file build_riscv.zig test
//! Run: whisper --isa rv32imfd --tohost 0x80001000 zig-out/bin/test_whisper

const blazt = @import("blazt");
const ops = blazt.ops;

// Whisper's tohost address for signaling completion
const TOHOST: *volatile u32 = @ptrFromInt(0x80001000);

// Signal test pass (exit code 0)
fn pass() noreturn {
    // Whisper convention: write 1 to tohost to signal successful exit
    TOHOST.* = 1;
    while (true) {
        asm volatile ("wfi");
    }
}

// Signal test fail with code
fn failWithCode(code: u32) noreturn {
    // Whisper convention: (exit_code << 1) | 1
    TOHOST.* = (code << 1) | 1;
    while (true) {
        asm volatile ("wfi");
    }
}

// Prevent optimizer from eliminating values
fn doNotOptimize(ptr: anytype) void {
    asm volatile (""
        :
        : [ptr] "r" (ptr),
        : "memory"
    );
}

// Volatile read to prevent compile-time evaluation
fn volatileRead(comptime T: type, ptr: *const T) T {
    return @as(*const volatile T, @ptrCast(ptr)).*;
}

// Approximate float equality check (noinline to prevent optimization)
fn approxEq(a: f32, b: f32, tolerance: f32) bool {
    @setRuntimeSafety(false);
    const diff = @abs(a - b);
    return diff <= tolerance;
}

// Test SAXPY: y = alpha * x + y
noinline fn testSaxpy() u32 {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    // Prevent compile-time evaluation
    doNotOptimize(&x);
    doNotOptimize(&y);

    const alpha: f32 = 2.0;

    // y = 2*x + y = [7, 10, 13, 16]
    ops.axpy(f32, 4, alpha, &x, &y);

    doNotOptimize(&y);

    if (!approxEq(y[0], 7.0, 0.01)) return 1;
    if (!approxEq(y[1], 10.0, 0.01)) return 2;
    if (!approxEq(y[2], 13.0, 0.01)) return 3;
    if (!approxEq(y[3], 16.0, 0.01)) return 4;

    return 0;
}

// Test SDOT: dot product
noinline fn testSdot() u32 {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    doNotOptimize(&x);
    doNotOptimize(&y);

    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 70
    const result = ops.dot(f32, &x, &y);

    if (!approxEq(result, 70.0, 0.01)) return 10;

    return 0;
}

// Test SSCAL: x = alpha * x
noinline fn testSscal() u32 {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    doNotOptimize(&x);

    const alpha: f32 = 3.0;

    // x = [3, 6, 9, 12]
    ops.scal(f32, 4, alpha, &x);

    doNotOptimize(&x);

    if (!approxEq(x[0], 3.0, 0.01)) return 20;
    if (!approxEq(x[1], 6.0, 0.01)) return 21;
    if (!approxEq(x[2], 9.0, 0.01)) return 22;
    if (!approxEq(x[3], 12.0, 0.01)) return 23;

    return 0;
}

// Test SNRM2: Euclidean norm
noinline fn testSnrm2() u32 {
    var x = [_]f32{ 3.0, 4.0 };

    doNotOptimize(&x);

    // sqrt(9 + 16) = 5
    const result = ops.nrm2(f32, &x);

    if (!approxEq(result, 5.0, 0.01)) return 30;

    return 0;
}

// Main test function (called after stack setup)
fn main() noreturn {
    var code: u32 = 0;

    code = testSaxpy();
    if (code != 0) failWithCode(code);

    code = testSdot();
    if (code != 0) failWithCode(code);

    code = testSscal();
    if (code != 0) failWithCode(code);

    code = testSnrm2();
    if (code != 0) failWithCode(code);

    // All tests passed
    pass();
}

// Entry point - initialize stack, enable FPU, then call main
// Stack at 0x00100000 (in the lower 16MB, within Whisper's default memory)
export fn _start() callconv(.naked) noreturn {
    // Set up stack
    // Enable FPU by setting mstatus.FS = 01 (Initial state)
    // mstatus.FS is bits [14:13], so we set 0x2000 (bit 13)
    asm volatile (
        \\lui sp, 0x100
        \\li t0, 0x2000
        \\csrs mstatus, t0
        \\jal zero, %[main]
        :
        : [main] "i" (&main),
    );
    unreachable;
}

// Panic handler for freestanding
pub fn panic(msg: []const u8, stack_trace: ?*@import("std").builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = msg;
    _ = stack_trace;
    _ = ret_addr;
    failWithCode(255);
}
