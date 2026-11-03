#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/conda/envs/label-factory/bin:/opt/conda/bin:${PATH}"
export SAM6D_PYTHON="${SAM6D_PYTHON:-/opt/conda/envs/sam6d/bin/python}"
export SAM6D_BLENDER_PATH="${SAM6D_BLENDER_PATH:-/opt/blender}"

if [[ ! -x "${SAM6D_PYTHON}" ]]; then
    echo "SAM-6D Python was not found: ${SAM6D_PYTHON}" >&2
    exit 1
fi

exec "$@"

