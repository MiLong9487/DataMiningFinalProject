#!/usr/bin/env bash
# Convenience launcher: sets the NVIDIA driver lib path NixOS needs
# so PyTorch can find libcuda.so, then forwards args to uv run.
set -euo pipefail
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
exec uv run "$@"
