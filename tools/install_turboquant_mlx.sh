#!/usr/bin/env bash
# Install arozanov/turboquant-mlx into mlx_agent_env (TurboQuant KV + fused Metal attention).
#
# pip install git+https://... fails on Python 3.14+ with:
#   BackendUnavailable: Cannot import 'setuptools.backends._legacy'
# because upstream pyproject pins the removed legacy backend. This script clones,
# patches pyproject.toml to setuptools.build_meta, then pip install -e.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/mlx_agent_env"
REPO_URL="${TURBOQUANT_MLX_URL:-https://github.com/arozanov/turboquant-mlx.git}"
SRC="${TURBOQUANT_MLX_SRC:-${ROOT}/vendor/turboquant-mlx}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "Missing ${VENV}. Run: python3 -m venv mlx_agent_env"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel

mkdir -p "$(dirname "${SRC}")"
if [[ -d "${SRC}/.git" ]]; then
  git -C "${SRC}" fetch origin --prune
  git -C "${SRC}" pull --ff-only 2>/dev/null || true
else
  rm -rf "${SRC}"
  git clone --depth 1 "${REPO_URL}" "${SRC}"
fi

python "${ROOT}/tools/patch_turboquant_pyproject.py" "${SRC}"
python -m pip install -e "${SRC}"

python -c "from turboquant_mlx import apply_patch, make_adaptive_cache; print('turboquant_mlx OK:', apply_patch.__name__, make_adaptive_cache.__name__)"
echo "Done. Source: ${SRC} (vendor/ is gitignored). MLX_USE_TURBO_KV=0 disables TurboQuant in the agent."
