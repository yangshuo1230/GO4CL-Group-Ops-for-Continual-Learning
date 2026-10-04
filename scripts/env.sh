#!/usr/bin/env bash
# Shared scratch paths for GO4CL launchers (NFS; avoid system /tmp).
# Sourced by scripts/phase1/*.sh

_GO4CL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_WORKSPACE="$(cd "${_GO4CL_ROOT}/.." && pwd)"

export TMPDIR="${TMPDIR:-${_GO4CL_ROOT}/.tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${_WORKSPACE}/.cache/uv}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/torch/inductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${TMPDIR}/torch/cuda}"

mkdir -p "${TMPDIR}" "${UV_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"
