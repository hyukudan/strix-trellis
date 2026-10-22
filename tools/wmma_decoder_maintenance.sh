#!/usr/bin/env bash
set -euo pipefail
# The scheduler owns this read-only control path.
[[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
