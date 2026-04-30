#!/usr/bin/env bash
# Full pipeline for benchmark prep + smoke tests + benchmark run.
# Ground truth: graphar-bench/config/benchmark.yaml (override path with first argument).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_CONFIG="${1:-graphar-bench/config/benchmark.yaml}"

echo "Environment setup"
bash graphar-bench/scripts/00_setup.sh

echo "Build C++ GAR converter"
CPP_BUILD_DIR="graphar-bench/convert/build"
if [[ ! -f "${CPP_BUILD_DIR}/ogbn_to_gar" ]]; then
  mkdir -p "${CPP_BUILD_DIR}"
  cmake -S graphar-bench/convert -B "${CPP_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "${CPP_BUILD_DIR}" --parallel "$(nproc)"

echo "Download dataset (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python graphar-bench/scripts/01_download.py --config "${BENCHMARK_CONFIG}"

echo "Convert dataset to GAR (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python graphar-bench/scripts/02_load_gar.py --config "${BENCHMARK_CONFIG}"

echo "Load dataset into Neo4j (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python graphar-bench/scripts/03_load_neo4j.py --config "${BENCHMARK_CONFIG}"

echo "Create Neo4j index"
bash graphar-bench/scripts/03b_neo4j_index.sh

echo "Configure Neo4j"
bash graphar-bench/scripts/03c_neo4j_conf.sh

echo "Verify GAR and Neo4j (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python graphar-bench/scripts/04_verify_formats.py --config "${BENCHMARK_CONFIG}"

echo "Smoke test: GAR training loop"
.venv/bin/python graphar-bench/scripts/05_smoke_train_gar.py --config "${BENCHMARK_CONFIG}"

echo "Smoke test: Neo4j loaders"
.venv/bin/python graphar-bench/scripts/05_smoke_train_neo4j.py --config "${BENCHMARK_CONFIG}"

DATASET=$(grep '^dataset:' "${BENCHMARK_CONFIG}" | awk '{print $2}')
RESULT_DIR="graphar-bench/results/${DATASET}/$(date +%Y%m%d-%H%M)"
mkdir -p "${RESULT_DIR}"

echo "Run benchmark (result dir: ${RESULT_DIR})"
# pip install -e "./python[ml]" # to update only graphar after git checkout
sudo perf record -g -F 99 --call-graph dwarf \
  .venv/bin/python graphar-bench/scripts/04_run_benchmark.py --config "${BENCHMARK_CONFIG}" --result-dir "${RESULT_DIR}"
sudo perf report --stdio --no-source --no-inline \
  > "${RESULT_DIR}/gar_perf.txt" 2> "${RESULT_DIR}/gar_perf.err"
sudo perf script --no-inline 2> "${RESULT_DIR}/perf_script.err" \
  | ./FlameGraph/stackcollapse-perf.pl > "${RESULT_DIR}/out.folded"
./FlameGraph/flamegraph.pl "${RESULT_DIR}/out.folded" > "${RESULT_DIR}/flamegraph.svg"

echo "Analyze results"
sudo .venv/bin/python graphar-bench/analyze.py "${RESULT_DIR}" --plots