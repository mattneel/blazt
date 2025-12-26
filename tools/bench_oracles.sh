#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
bench_oracles.sh — run blazt benchmarks vs available oracle BLAS libraries.

Defaults:
  - Runs two full bench passes:
    1) Oracles forced single-thread (NUM_THREADS=1)
    2) Oracles set to max threads (NUM_THREADS=nproc)
  - Writes full stdout logs into ./bench_results/ (gitignored)
  - Prints a compact p50 GFLOP/s summary for GEMM-related benches

Environment / knobs:
  BLAZT_BENCH_LAPACK_HEAVY=1    Use heavier LAPACK bench sizes (slow)
  BLAZT_BENCH_ORACLE=1          Enable oracle comparison in the bench
  BLAZT_ORACLE_BLIS=<path>      Explicit BLIS shared library path (optional)
  BLAZT_ORACLE_MKL=<path>       Explicit MKL shared library path (optional; prefer libmkl_rt.so)

Options:
  --threads N    Override the "max threads" pass (default: nproc)
  --no-build-blis  Do not attempt to build BLIS if not found
  --out DIR      Output directory (default: ./bench_results)
  -h, --help     Show help

Examples:
  ./tools/bench_oracles.sh
  BLAZT_BENCH_LAPACK_HEAVY=1 ./tools/bench_oracles.sh --threads 32
EOF
}

max_threads=""
out_dir=""
build_blis=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      max_threads="${2:-}"
      shift 2
      ;;
    --out)
      out_dir="${2:-}"
      shift 2
      ;;
    --no-build-blis)
      build_blis=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown arg: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ -z "${out_dir}" ]]; then
  out_dir="${repo_root}/bench_results"
fi
mkdir -p "${out_dir}"

if [[ -z "${max_threads}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    max_threads="$(nproc)"
  else
    max_threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  fi
fi

if ! [[ "${max_threads}" =~ ^[0-9]+$ ]] || [[ "${max_threads}" -lt 1 ]]; then
  echo "error: --threads must be a positive integer (got: ${max_threads})" >&2
  exit 2
fi

export BLAZT_BENCH_ORACLE=1

blis_lib="${BLAZT_ORACLE_BLIS:-}"
if [[ -z "${blis_lib}" ]]; then
  blis_lib="$(find "${repo_root}/oracle-build/blis" -maxdepth 4 -type f -name 'libblis.so*' -print -quit 2>/dev/null || true)"
fi

if [[ -z "${blis_lib}" && "${build_blis}" -eq 1 ]]; then
  if [[ -x "${repo_root}/deps/blis/configure" ]]; then
    echo "info: BLIS not found; building out-of-tree under oracle-build/blis" >&2
    mkdir -p "${repo_root}/oracle-build/blis"
    (
      cd "${repo_root}/oracle-build/blis"
      if [[ ! -f config.mk ]]; then
        ../../deps/blis/configure auto
      fi
      make -j"${max_threads}"
    )
    blis_lib="$(find "${repo_root}/oracle-build/blis" -maxdepth 4 -type f -name 'libblis.so*' -print -quit 2>/dev/null || true)"
  fi
fi

if [[ -n "${blis_lib}" ]]; then
  export BLAZT_ORACLE_BLIS="${blis_lib}"
  echo "info: using BLIS: ${blis_lib}" >&2
else
  echo "info: BLIS not available (set BLAZT_ORACLE_BLIS=/path/to/libblis.so or allow build)" >&2
fi

# MKL on some systems requires an OpenMP runtime to be present at load time. Prefer libgomp.
if [[ -f /lib64/libgomp.so.1 ]]; then
  export LD_PRELOAD="/lib64/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi

run_bench() {
  local label="$1"
  local thr="$2"
  local out="${out_dir}/full_${label}.txt"

  echo "==> zig build bench (oracle threads=${thr}) -> ${out}" >&2
  OPENBLAS_NUM_THREADS="${thr}" \
  BLIS_NUM_THREADS="${thr}" \
  MKL_NUM_THREADS="${thr}" \
  OMP_NUM_THREADS="${thr}" \
    zig build bench > "${out}"
}

run_bench "single_thread" 1
run_bench "${max_threads}_threads" "${max_threads}"

summarize_file() {
  local path="$1"
  awk '
    /^bench / { name = substr($0, 7); next }
    /^  p50:/ {
      if (name ~ /^(ops\.gemm\(f32,row_major\)|parallel\.gemm\(f32,row_major,threads=|ops\.gemm\(f32,col_major\)|ops\.gemm\(f32,col_major,oracle_cmp\)|oracle\.sgemm\()/) {
        if (match($0, /\(([0-9.]+) GFLOP\/s\)/, m)) {
          printf("%-45s %s\n", name, m[1]);
        }
      }
      next
    }
  ' "${path}"
}

echo
echo "=== GEMM/oracle p50 GFLOP/s summary (single-thread oracles) ==="
summarize_file "${out_dir}/full_single_thread.txt"
echo
echo "=== GEMM/oracle p50 GFLOP/s summary (oracles threads=${max_threads}) ==="
summarize_file "${out_dir}/full_${max_threads}_threads.txt"


