#!/usr/bin/env bash
# One-time setup of a multi-GPU node (written for 8x B300, works on any recent NVIDIA box).
#
#   bash setup_b300.sh            # then: bash launch_bt4.sh
#
# Does, idempotently:
#   1. uv + Python env from uv.lock, with the PyTorch CUDA build matching the driver
#      (cu130 for CUDA-13 drivers, which B300 / sm_103 wants; cu128 otherwise)
#   2. Stockfish (used by the puzzle-rate evaluation during training)
#   3. Leela BT4 weights (BT4-1024x15x32h-swa-6147500), sha256-checked
#   4. Lichess puzzle dataset -> src/dataset/with_best_move/{trainset,testset}.pt
#   5. BT4 self-check (port vs recorded lc0 outputs, adapter vs BT4 policy)
#
# Environment overrides:
#   BT4_WEIGHTS=/path/to/BT4.pb.gz   use a local copy instead of downloading
#   TORCH_CUDA=cu130|cu128|cu126     force the PyTorch CUDA build
#   SKIP_DATASET=1 / SKIP_STOCKFISH=1 / SKIP_CHECK=1
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

BT4_URL="https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz"
BT4_SHA256="66e6c1b5308c34230801dc171df580d7a66ec916dc2211246a485ef8e545c67e"
BT4_DEST="$ROOT/weights/BT4-1024x15x32h-swa-6147500.pb.gz"
STOCKFISH_TAG="sf_17.1"
TORCH_VERSION="2.9.1"

log() { echo -e "\n=== $* ==="; }

# ---------------------------------------------------------------- 0. sanity
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: is this a GPU node?"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
for tool in git curl make g++; do
  command -v $tool >/dev/null || { echo "missing '$tool' (needed for setup); install it first"; exit 1; }
done

# ---------------------------------------------------------------- 1. python env
log "Python environment"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen

if [ -z "${TORCH_CUDA:-}" ]; then
  drv_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  if [ "${drv_cuda:-0}" -ge 130 ]; then TORCH_CUDA=cu130
  elif [ "${drv_cuda:-0}" -ge 128 ]; then TORCH_CUDA=cu128
  else TORCH_CUDA=cu126; fi
fi
echo "driver supports CUDA ${drv_cuda:-?}; installing torch ${TORCH_VERSION}+${TORCH_CUDA}"
# NB: a later plain `uv sync` would put the PyPI build back; use `uv run --no-sync` or .venv/bin/python.
uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchaudio \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
  "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPUs"
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
print(f"torch {torch.__version__} cuda {torch.version.cuda}: {torch.cuda.device_count()} GPUs, capabilities {caps}, "
      f"archs {torch.cuda.get_arch_list()}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()   # fails here if the wheel has no kernels for this GPU
EOF

# ---------------------------------------------------------------- 2. stockfish
if [ -z "${SKIP_STOCKFISH:-}" ] && [ ! -x Stockfish/src/stockfish ]; then
  log "Stockfish ${STOCKFISH_TAG}"
  rm -rf Stockfish
  git clone --depth 1 --branch "$STOCKFISH_TAG" https://github.com/official-stockfish/Stockfish.git
  (cd Stockfish/src && make -j"$(nproc)" build ARCH=native >/dev/null)
fi
[ -x Stockfish/src/stockfish ] && echo "stockfish: $(echo quit | Stockfish/src/stockfish | head -1)"

# ---------------------------------------------------------------- 3. BT4 weights
log "BT4 weights"
mkdir -p weights
if [ -n "${BT4_WEIGHTS:-}" ]; then
  cp -n "$BT4_WEIGHTS" "$BT4_DEST" || true
elif [ ! -f "$BT4_DEST" ]; then
  curl -fL --retry 5 -o "$BT4_DEST.part" "$BT4_URL" && mv "$BT4_DEST.part" "$BT4_DEST"
fi
echo "$BT4_SHA256  $BT4_DEST" | sha256sum -c -

# ---------------------------------------------------------------- 4. dataset
if [ -z "${SKIP_DATASET:-}" ] && [ ! -f src/dataset/with_best_move/trainset.pt ]; then
  log "Lichess puzzle dataset"
  .venv/bin/python src/tokenization/prepare_dataset.py
fi
ls -la src/dataset/with_best_move/ 2>/dev/null || true

# ---------------------------------------------------------------- 5. self-check
if [ -z "${SKIP_CHECK:-}" ]; then
  log "BT4 self-check"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/BT4/check_bt4.py --weights "$BT4_DEST"
fi

log "Setup done. Launch with: bash launch_bt4.sh"
