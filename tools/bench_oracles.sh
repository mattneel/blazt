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
  --physical     Use physical cores only (one CPU per core) for pinning; defaults --threads to physical-core count
  --no-build-blis  Do not attempt to build BLIS if not found
  --out DIR      Output directory (default: ./bench_results)
  --no-pin       Do not pin the benchmark process to a fixed CPU set
  --cpu-list LIST  CPU list for pinning (taskset -c LIST). Default: 0-(threads-1)
  -h, --help     Show help

Examples:
  ./tools/bench_oracles.sh
  BLAZT_BENCH_LAPACK_HEAVY=1 ./tools/bench_oracles.sh --threads 32
EOF
}

max_threads=""
out_dir=""
build_blis=1
no_pin=0
cpu_list=""
physical=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      max_threads="${2:-}"
      shift 2
      ;;
    --physical)
      physical=1
      shift
      ;;
    --out)
      out_dir="${2:-}"
      shift 2
      ;;
    --no-build-blis)
      build_blis=0
      shift
      ;;
    --no-pin)
      no_pin=1
      shift
      ;;
    --cpu-list)
      cpu_list="${2:-}"
      shift 2
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
  if [[ "${physical}" -eq 1 ]] && command -v lscpu >/dev/null 2>&1; then
    max_threads="$(
      lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, '
        $0 ~ /^#/ { next }
        {
          key = $3 "-" $2
          if (!(key in seen)) { seen[key] = 1; count++ }
        }
        END { if (count > 0) print count; else print 1 }
      '
    )"
  else
    if command -v nproc >/dev/null 2>&1; then
      max_threads="$(nproc)"
    else
      max_threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
  fi
fi

if ! [[ "${max_threads}" =~ ^[0-9]+$ ]] || [[ "${max_threads}" -lt 1 ]]; then
  echo "error: --threads must be a positive integer (got: ${max_threads})" >&2
  exit 2
fi

export BLAZT_BENCH_ORACLE=1

# Disable dynamic thread behavior for consistency across runs.
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

# If taskset exists and pinning isn't disabled, pin the whole benchmark process to a stable CPU set.
# This keeps all threads (including oracles) on the same CPUs across runs.
taskset_cmd=()
if [[ "${no_pin}" -eq 0 ]] && command -v taskset >/dev/null 2>&1; then
  if [[ -z "${cpu_list}" ]]; then
    if [[ "${physical}" -eq 1 ]] && command -v lscpu >/dev/null 2>&1; then
      # Choose one logical CPU per physical core (first thread in each core).
      cpu_list="$(
        lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, -v want="${max_threads}" '
          $0 ~ /^#/ { next }
          {
            cpu = $1; core = $2; sock = $3;
            key = sock "-" core;
            if (!(key in seen)) {
              seen[key] = 1;
              cpus[count++] = cpu;
              if (count >= want) exit;
            }
          }
          END {
            if (count == 0) { print "0"; exit }
            for (i = 0; i < count; i++) {
              printf "%s%s", cpus[i], (i + 1 < count ? "," : "")
            }
          }
        '
      )"
    else
      cpu_list="0-$((max_threads - 1))"
    fi
  fi
  taskset_cmd=(taskset -c "${cpu_list}")
  echo "info: pinning benchmark process with taskset -c ${cpu_list}" >&2
else
  if [[ "${no_pin}" -eq 0 ]]; then
    echo "info: taskset not found; running without CPU pinning (use --no-pin to silence)" >&2
  fi
fi

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
    "${taskset_cmd[@]}" zig build bench > "${out}"
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

fmt1() {
  local v="${1:-}"
  if [[ -z "${v}" ]]; then
    echo "n/a"
    return 0
  fi
  # Print with 1 decimal when it's numeric, otherwise pass through.
  if [[ "${v}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf "%.1f" "${v}"
  else
    echo "${v}"
  fi
}

p50_gflops() {
  local path="$1"
  local want="$2"
  awk -v want="$want" '
    /^bench / { name = substr($0, 7); next }
    /^  p50:/ && name == want {
      if (match($0, /\(([0-9.]+) GFLOP\/s\)/, m)) { print m[1]; exit }
    }
  ' "${path}"
}

bold_if() {
  local label="$1"
  local best="$2"
  local val="$3"
  if [[ "${label}" == "${best}" ]]; then
    printf "**%s**" "${val}"
  else
    printf "%s" "${val}"
  fi
}

best_label() {
  local b="$1"
  local o="$2"
  local l="$3"
  local m="$4"
  awk -v b="$b" -v o="$o" -v l="$l" -v m="$m" '
    function ok(x) { return x ~ /^[0-9]+(\.[0-9]+)?$/ }
    function num(x) { return x + 0.0 }
    BEGIN {
      max = -1; lab = "";
      if (ok(b) && num(b) > max) { max = num(b); lab = "blazt" }
      if (ok(o) && num(o) > max) { max = num(o); lab = "OpenBLAS" }
      if (ok(l) && num(l) > max) { max = num(l); lab = "BLIS" }
      if (ok(m) && num(m) > max) { max = num(m); lab = "MKL" }
      print lab
    }
  '
}

echo
echo "=== GEMM/oracle p50 GFLOP/s summary (single-thread oracles) ==="
summarize_file "${out_dir}/full_single_thread.txt"
echo
echo "=== GEMM/oracle p50 GFLOP/s summary (oracles threads=${max_threads}) ==="
summarize_file "${out_dir}/full_${max_threads}_threads.txt"

single_file="${out_dir}/full_single_thread.txt"
max_file="${out_dir}/full_${max_threads}_threads.txt"

blazt_1="$(p50_gflops "${single_file}" "ops.gemm(f32,row_major)")"
blazt_n="$(p50_gflops "${max_file}" "parallel.gemm(f32,row_major,threads=${max_threads})")"

openblas_1="$(p50_gflops "${single_file}" "oracle.sgemm(openblas,col_major)")"
openblas_n="$(p50_gflops "${max_file}" "oracle.sgemm(openblas,col_major)")"

blis_1="$(p50_gflops "${single_file}" "oracle.sgemm(blis,col_major)")"
blis_n="$(p50_gflops "${max_file}" "oracle.sgemm(blis,col_major)")"

mkl_1="$(p50_gflops "${single_file}" "oracle.sgemm(mkl,col_major)")"
mkl_n="$(p50_gflops "${max_file}" "oracle.sgemm(mkl,col_major)")"

best_n="$(best_label "${blazt_n}" "${openblas_n}" "${blis_n}" "${mkl_n}")"

echo
echo "=== Markdown snippet ==="
if [[ -n "${mkl_1}" ]]; then
  echo "Now MKL is behaving ($(fmt1 "${mkl_1}") GFLOP/s single-threaded)."
fi
echo
echo "**${max_threads} threads, all libraries:**"
echo '```'
printf "blazt:     %s GFLOP/s\n" "$(fmt1 "${blazt_n}")"
printf "OpenBLAS:  %s GFLOP/s\n" "$(fmt1 "${openblas_n}")"
printf "BLIS:      %s GFLOP/s\n" "$(fmt1 "${blis_n}")"
printf "MKL:       %s GFLOP/s\n" "$(fmt1 "${mkl_n}")"
echo '```'
echo
echo "| Threads | blazt | OpenBLAS | BLIS | MKL |"
echo "|---------|-------|----------|------|-----|"
printf "| 1 | %s | %s | %s | %s |\n" \
  "$(fmt1 "${blazt_1}")" \
  "$(fmt1 "${openblas_1}")" \
  "$(fmt1 "${blis_1}")" \
  "$(fmt1 "${mkl_1}")"
printf "| %d | %s | %s | %s | %s |\n" \
  "${max_threads}" \
  "$(bold_if "blazt" "${best_n}" "$(fmt1 "${blazt_n}")")" \
  "$(bold_if "OpenBLAS" "${best_n}" "$(fmt1 "${openblas_n}")")" \
  "$(bold_if "BLIS" "${best_n}" "$(fmt1 "${blis_n}")")" \
  "$(bold_if "MKL" "${best_n}" "$(fmt1 "${mkl_n}")")"


