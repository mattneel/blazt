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
  BLAZT_ORACLE_OPENBLAS=<path>  Explicit OpenBLAS shared library path (optional)
  BLAZT_ORACLE_BLIS=<path>      Explicit BLIS shared library path (optional)
  BLAZT_ORACLE_MKL=<path>       Explicit MKL shared library path (optional; prefer libmkl_rt.so)

Options:
  --threads N    Override the "max threads" pass (default: nproc)
  --physical     Use physical cores only (one CPU per core) for pinning; defaults --threads to physical-core count
  --p-only       Like --physical, but only selects cores that have SMT siblings (often P-cores on hybrid Intel)
  --no-build-blis  Do not attempt to build BLIS if not found
  --no-build-openblas  Do not attempt to build OpenBLAS if not found
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
max_threads_user=0
out_dir=""
build_blis=1
build_openblas=1
no_pin=0
cpu_list=""
physical=0
p_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      max_threads="${2:-}"
      max_threads_user=1
      shift 2
      ;;
    --physical)
      physical=1
      shift
      ;;
    --p-only)
      physical=1
      p_only=1
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
    --no-build-openblas)
      build_openblas=0
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
    if [[ "${p_only}" -eq 1 ]]; then
      max_threads="$(
        lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, '
          $0 ~ /^#/ { next }
          { key = $3 "-" $2; core_n[key]++ }
          END {
            p = 0; all = 0;
            for (k in core_n) { all++; if (core_n[k] >= 2) p++; }
            if (p > 0) print p; else if (all > 0) print all; else print 1;
          }
        '
      )"
    else
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
    fi
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
        lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, -v want="${max_threads}" -v p_only="${p_only}" '
          $0 ~ /^#/ { next }
          {
            cpu = $1; core = $2; sock = $3;
            key = sock "-" core;
            n = core_n[key] + 0;
            core_cpu[key, n] = cpu;
            core_n[key] = n + 1;
            if (!(core_seen[key])) { core_seen[key] = 1; core_order[core_count++] = key; }
          }
          END {
            if (core_count == 0) { print "0"; exit }
            out_n = 0;
            for (oi = 0; oi < core_count; oi++) {
              key = core_order[oi];
              n = core_n[key] + 0;
              if (p_only == 1 && n < 2) continue;
              cpus[out_n++] = core_cpu[key, 0];
              if (out_n >= want) break;
            }
            if (out_n == 0) {
              # Fallback if SMT-only selection yielded nothing.
              for (oi = 0; oi < core_count; oi++) {
                key = core_order[oi];
                cpus[out_n++] = core_cpu[key, 0];
                if (out_n >= want) break;
              }
            }
            for (i = 0; i < out_n; i++) {
              printf "%s%s", cpus[i], (i + 1 < out_n ? "," : "")
            }
          }
        '
      )"
    else
      cpu_list="0-$((max_threads - 1))"
    fi
  fi

  if [[ "${physical}" -eq 1 ]] && [[ "${cpu_list}" == *,* ]]; then
    pin_count="$(echo "${cpu_list}" | awk -F, '{print NF}')"
    if [[ "${max_threads_user}" -eq 0 ]]; then
      max_threads="${pin_count}"
    else
      if [[ "${max_threads}" -gt "${pin_count}" ]]; then
        echo "error: requested --threads=${max_threads} but cpu list only has ${pin_count} CPUs (${cpu_list})" >&2
        exit 2
      fi
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

# If we have a local BLIS build but it's configured single-threaded, reconfigure it to pthreads.
if [[ -n "${blis_lib}" && "${build_blis}" -eq 1 ]]; then
  if [[ "${blis_lib}" == "${repo_root}/oracle-build/blis/"* ]] && [[ -f "${repo_root}/oracle-build/blis/config.mk" ]] && [[ -x "${repo_root}/deps/blis/configure" ]]; then
    if grep -Eq '^THREADING_MODEL[[:space:]]*:=[[:space:]]*single' "${repo_root}/oracle-build/blis/config.mk"; then
      echo "info: rebuilding BLIS with pthreads threading (local build was single-threaded)" >&2
      (
        cd "${repo_root}/oracle-build/blis"
        ../../deps/blis/configure -t pthreads auto
        make -j"${max_threads}"
      )
      blis_lib="$(find "${repo_root}/oracle-build/blis" -maxdepth 4 -type f -name 'libblis.so*' -print -quit 2>/dev/null || true)"
    fi
  fi
fi

if [[ -z "${blis_lib}" && "${build_blis}" -eq 1 ]]; then
  if [[ -x "${repo_root}/deps/blis/configure" ]]; then
    echo "info: BLIS not found; building out-of-tree under oracle-build/blis" >&2
    mkdir -p "${repo_root}/oracle-build/blis"
    (
      cd "${repo_root}/oracle-build/blis"
      need_cfg=0
      if [[ ! -f config.mk ]]; then
        need_cfg=1
      else
        if ! grep -Eq '^THREADING_MODEL[[:space:]]*:=[[:space:]]*.*pthreads' config.mk; then
          need_cfg=1
        fi
      fi
      if [[ "${need_cfg}" -eq 1 ]]; then
        # Force pthreads threading so BLIS_NUM_THREADS actually scales.
        ../../deps/blis/configure -t pthreads auto
      fi
      make -j"${max_threads}"
    )
    blis_lib="$(find "${repo_root}/oracle-build/blis" -maxdepth 4 -type f -name 'libblis.so*' -print -quit 2>/dev/null || true)"
  fi
fi

if [[ -n "${blis_lib}" ]]; then
  export BLAZT_ORACLE_BLIS="${blis_lib}"
  export BLIS_THREAD_IMPL="${BLIS_THREAD_IMPL:-pthreads}"
  echo "info: using BLIS: ${blis_lib}" >&2
else
  echo "info: BLIS not available (set BLAZT_ORACLE_BLIS=/path/to/libblis.so or allow build)" >&2
fi

openblas_lib="${BLAZT_ORACLE_OPENBLAS:-}"
if [[ -z "${openblas_lib}" ]]; then
openblas_lib="$(find "${repo_root}/oracle-build/openblas" -maxdepth 4 -type f \( -name 'libopenblas*.so*' -o -name 'libopenblasp*.so*' \) -print -quit 2>/dev/null || true)"
fi

if [[ -z "${openblas_lib}" && "${build_openblas}" -eq 1 ]]; then
  if [[ -d "${repo_root}/deps/openblas" ]]; then
    echo "info: OpenBLAS not found; building under oracle-build/openblas (pthreads)" >&2
    mkdir -p "${repo_root}/oracle-build/openblas"
    (
      cd "${repo_root}/oracle-build/openblas"
      if [[ ! -d src ]]; then
        mkdir -p src
        cp -a "${repo_root}/deps/openblas/." src/
      fi
      cd src
      make -j"${max_threads}" DYNAMIC_ARCH=1 USE_THREAD=1 NUM_THREADS="${max_threads}" NO_AFFINITY=1 shared
    )
openblas_lib="$(find "${repo_root}/oracle-build/openblas/src" -maxdepth 1 -type f \( -name 'libopenblas*.so*' -o -name 'libopenblasp*.so*' \) -print -quit 2>/dev/null || true)"
  fi
fi

if [[ -n "${openblas_lib}" ]]; then
  export BLAZT_ORACLE_OPENBLAS="${openblas_lib}"
  # Disable any internal OpenBLAS affinity; we pin externally via taskset.
  export OPENBLAS_AFFINITY="${OPENBLAS_AFFINITY:-0}"
  echo "info: using OpenBLAS: ${openblas_lib}" >&2
else
  echo "info: OpenBLAS not available (set BLAZT_ORACLE_OPENBLAS=/path/to/libopenblas.so or allow build)" >&2
fi

# MKL on some systems requires an OpenMP runtime to be present at load time.
#
# Prefer Intel OpenMP if available (best match for MKL), otherwise fall back to GNU OpenMP.
iomp_path=""
for p in \
  /opt/intel/oneapi/compiler/*/lib/libiomp5.so \
  /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so \
  ; do
  if [[ -f "${p}" ]]; then
    iomp_path="${p}"
    break
  fi
done

if [[ -n "${iomp_path}" ]]; then
  export LD_PRELOAD="${iomp_path}${LD_PRELOAD:+:$LD_PRELOAD}"
  export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-INTEL}"
elif [[ -f /lib64/libgomp.so.1 ]]; then
  export LD_PRELOAD="/lib64/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
  export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
fi

# Avoid OpenMP runtimes trying to (re)bind threads; we handle affinity via taskset.
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"

run_bench() {
  local label="$1"
  local thr="$2"
  local out="${out_dir}/full_${label}.txt"

  echo "==> zig build bench (oracle threads=${thr}) -> ${out}" >&2
  local this_taskset=("${taskset_cmd[@]}")
  if [[ "${no_pin}" -eq 0 ]] && [[ "${thr}" -eq 1 ]] && [[ -n "${cpu_list}" ]] && command -v taskset >/dev/null 2>&1; then
    # For single-thread runs, pin to ONE CPU (first in list) so we can't accidentally land on an E-core.
    local first_cpu
    first_cpu="$(echo "${cpu_list}" | awk -F, '{print $1}')"
    this_taskset=(taskset -c "${first_cpu}")
    echo "info: single-thread pass pinned to CPU ${first_cpu}" >&2
  fi
  OPENBLAS_NUM_THREADS="${thr}" \
  GOTO_NUM_THREADS="${thr}" \
  BLIS_NUM_THREADS="${thr}" \
  MKL_NUM_THREADS="${thr}" \
  OMP_NUM_THREADS="${thr}" \
    "${this_taskset[@]}" zig build bench > "${out}"
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

warn_if_no_scale() {
  local name="$1"
  local v1="$2"
  local vN="$3"
  awk -v name="$name" -v v1="$v1" -v vN="$vN" '
    function ok(x) { return x ~ /^[0-9]+(\.[0-9]+)?$/ }
    BEGIN {
      if (!ok(v1) || !ok(vN) || v1 == 0) exit 0;
      ratio = vN / v1;
      if (ratio < 1.10) {
        printf("warn: %s did not scale with threads (%.3f -> %.3f GFLOP/s, x%.2f). Likely single-thread build or threading disabled.\n", name, v1, vN, ratio) > "/dev/stderr";
      }
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

warn_if_no_scale "OpenBLAS" "${openblas_1}" "${openblas_n}"
warn_if_no_scale "BLIS" "${blis_1}" "${blis_n}"
warn_if_no_scale "MKL" "${mkl_1}" "${mkl_n}"

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


