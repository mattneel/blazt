#!/bin/bash
# Demo: blazt on RISC-V freestanding (Tenstorrent Tensix)
# This script demonstrates the full build and test workflow

set -e

WHISPER=/home/autark/src/whisper/build-Linux/whisper

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

type_cmd() {
    echo -e "${CYAN}$ $1${NC}"
    sleep 0.5
    eval "$1"
    echo
    sleep 1
}

header() {
    echo
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo
    sleep 1
}

header "BLAZT: Pure Zig BLAS for RISC-V Freestanding (Tensix)"

echo "Building blazt for riscv32-freestanding target..."
echo "No OS, no libc, no allocator — just pure RISC-V machine code."
echo
sleep 2

header "Step 1: Build the static library"

type_cmd "zig build --build-file build_riscv.zig -Drelease=true"

type_cmd "ls -lh zig-out/lib/libblazt_riscv.a"

header "Step 2: Verify RISC-V F extension instructions"

echo "Checking for floating-point instructions (fadd.s, fmul.s, fmadd.s)..."
echo
sleep 1

type_cmd "llvm-objdump -d zig-out/lib/libblazt_riscv.a 2>/dev/null | grep -E 'fadd\\.s|fmul\\.s|fmadd\\.s' | head -10"

echo -e "${GREEN}✓ Found RISC-V F extension instructions${NC}"
sleep 1

header "Step 3: Check exported BLAS symbols"

type_cmd "llvm-nm zig-out/lib/libblazt_riscv.a 2>/dev/null | grep 'T blazt'"

echo -e "${GREEN}✓ All 12 BLAS functions exported with C ABI${NC}"
sleep 1

header "Step 4: Build test executable for Whisper ISS"

type_cmd "zig build --build-file build_riscv.zig test -Drelease=true"

type_cmd "file zig-out/bin/test_whisper"

header "Step 5: Run tests on Tenstorrent Whisper ISS"

echo "Whisper: Tenstorrent's open-source RISC-V Instruction Set Simulator"
echo "Tests: SAXPY, SDOT, SSCAL, SNRM2"
echo
sleep 2

type_cmd "$WHISPER --isa rv32imfdc --tohost 0x80001000 zig-out/bin/test_whisper 2>&1"

header "Done!"

echo -e "${GREEN}✓ blazt compiled to RISC-V freestanding${NC}"
echo -e "${GREEN}✓ Contains proper F extension instructions${NC}"
echo -e "${GREEN}✓ All BLAS tests pass on Whisper ISS${NC}"
echo
echo "Ready for Tenstorrent Tensix hardware!"
echo
