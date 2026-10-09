#!/usr/bin/env bash
# CPU smoke for the layer-wise parameter patch. Does not train.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=""
exec uv run go4cl phase2 param-patch "$@"
