#!/usr/bin/env bash
# Clone MLX (and optionally mlx-lm) from your forks, then editable-install into mlx_agent_env.
# On Apple Silicon, the C++ extension is built with Metal by default; CMAKE_ARGS can override.
#
# Usage:
#   export MLX_REPO_URL="https://github.com/YOU/mlx.git"
#   export MLX_REPO_REF="main"                    # optional branch / tag
#   export MLX_LM_REPO_URL="https://github.com/YOU/mlx-lm.git"   # optional; skip to keep PyPI mlx-lm
#   export MLX_LM_REPO_REF="main"
#   ./tools/install_mlx_fork_metal.sh
#
# Requires: Xcode Command Line Tools (clang), git, Python 3 venv at repo/mlx_agent_env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/mlx_agent_env"
VENDOR="${ROOT}/vendor"

MLX_URL="${MLX_REPO_URL:-https://github.com/ml-explore/mlx.git}"
MLX_REF="${MLX_REPO_REF:-}"
MLX_LM_URL="${MLX_LM_REPO_URL:-}"
MLX_LM_REF="${MLX_LM_REPO_REF:-}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "Missing ${VENV}. Create it first: python3 -m venv mlx_agent_env"
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This workflow is for macOS (Metal). For other platforms see https://github.com/ml-explore/mlx"
  exit 1
fi

if ! xcode-select -p &>/dev/null; then
  echo "Install Xcode Command Line Tools: xcode-select --install"
  exit 1
fi

# Ensure Metal-enabled MLX build unless the caller already set CMAKE_ARGS.
if [[ -z "${CMAKE_ARGS:-}" ]]; then
  export CMAKE_ARGS="-DMLX_BUILD_METAL=ON"
else
  export CMAKE_ARGS
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "cmake>=3.24" ninja

mkdir -p "${VENDOR}"

clone_or_update() {
  local url="$1"
  local dir="$2"
  local ref="${3:-}"
  if [[ -d "${dir}/.git" ]]; then
    git -C "${dir}" fetch origin --prune
  else
    mkdir -p "$(dirname "${dir}")"
    git clone "${url}" "${dir}"
  fi
  if [[ -n "${ref}" ]]; then
    git -C "${dir}" checkout "${ref}"
  fi
  git -C "${dir}" pull --ff-only 2>/dev/null || true
}

echo "==> MLX from ${MLX_URL}"
clone_or_update "${MLX_URL}" "${VENDOR}/mlx" "${MLX_REF}"
echo "==> pip install -e (Metal build; may take several minutes)"
python -m pip install -e "${VENDOR}/mlx"

if [[ -n "${MLX_LM_URL}" ]]; then
  echo "==> mlx-lm from ${MLX_LM_URL}"
  clone_or_update "${MLX_LM_URL}" "${VENDOR}/mlx-lm" "${MLX_LM_REF}"
  python -m pip install -e "${VENDOR}/mlx-lm"
else
  echo "==> Skipping mlx-lm fork (MLX_LM_REPO_URL unset). Install/upgrade from PyPI if needed:"
  echo "    pip install 'mlx-lm>=0.30.0'"
fi

echo ""
echo "Verify Metal path is active:"
python - <<'PY'
import mlx.core as mx
print("mlx version:", getattr(mx, "__version__", "unknown"))
a = mx.array([1.0, 2.0])
print("default device:", mx.default_device())
print("add ok:", (a + 1).tolist())
PY

echo ""
echo "Done. Editable installs live under ${VENDOR}/ (gitignored)."
