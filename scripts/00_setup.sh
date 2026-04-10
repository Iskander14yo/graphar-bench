#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
VENV_PIP="${ROOT_DIR}/.venv/bin/pip"

install_neo4j_if_missing() {
  if command -v neo4j >/dev/null 2>&1 && command -v neo4j-admin >/dev/null 2>&1; then
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Neo4j."
    exit 1
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Automatic Neo4j install currently supports Debian/Ubuntu (apt-get)."
    exit 1
  fi

  echo "Installing Neo4j via apt..."
  sudo apt-get update
  sudo apt-get install -y wget gnupg ca-certificates
  sudo install -d -m 0755 /etc/apt/keyrings
  wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor --batch --yes -o /etc/apt/keyrings/neotechnology.gpg
  echo "deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y neo4j
}

install_arrow_if_missing() {
  wget https://packages.apache.org/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
  sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
  sudo apt update
  sudo apt install -y -V libarrow-dev libarrow-dataset-dev libarrow-acero-dev libparquet-dev
}

sudo apt-get update
sudo apt install -y python3-pip python3-venv build-essential
install_arrow_if_missing

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Creating virtualenv at .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

export CXXFLAGS="-g -fno-omit-frame-pointer"  # for perf
pip install -e "./python[ml-benchmark]"

# torch-sparse is required by PyG's NeighborLoader; pick CPU or CUDA wheel
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  TORCH_CUDA_TAG="cu$(nvidia-smi --query-gpu=driver_model.current --format=csv,noheader 2>/dev/null | head -1 || echo '')"
  TORCH_VERSION=$(python -c "import torch; v=torch.__version__; print(v.split('+')[0])")
  CUDA_VERSION=$(python -c "import torch; cv=torch.version.cuda; print('cu'+''.join(cv.split('.')))" 2>/dev/null || echo "cpu")
  PYG_TORCH_TAG="torch-${TORCH_VERSION}+${CUDA_VERSION}"
else
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  PYG_TORCH_TAG="torch-${TORCH_VERSION}+cpu"
fi
pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/${PYG_TORCH_TAG}.html"

install_neo4j_if_missing

# install kernel symbols (for perf and flamegraph)
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r) libc6-dbg -y

if ! command -v neo4j >/dev/null 2>&1 || ! command -v neo4j-admin >/dev/null 2>&1; then
  echo "Neo4j installation failed or binaries are not in PATH."
  exit 1
fi

echo "Setup OK: Python deps installed, Neo4j installed."
