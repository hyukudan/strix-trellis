#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/lowbit-core" ]]
[[ ! -e frozen.sha256 ]]
sha256sum build/v9.a | grep '^1e111789526692f0b9a3e713fcbf6723a1bec4d959a1a24cb9dde175001b0325 '
find src bench tools fixtures build -type f -print0 | sort -z | xargs -0 sha256sum > frozen.sha256
sha256sum -c --quiet frozen.sha256
