#!/usr/bin/env bash
# Shared by cluster entry points. This subtree is reserved for this worktree.
trellis_die() { echo "strix-trellis: $*" >&2; exit 1; }
TRELLIS_OWN_ROOT=$(realpath -m -- "$HOME/gpu-site/scratch-trellis")
# Reject a symlink redirect of any ancestor outside the permitted scratch tree.
[[ "$TRELLIS_OWN_ROOT" == "$HOME/gpu-site/scratch-trellis" ]] ||
    trellis_die "Owned scratch path must not redirect through symlinks"
trellis_require_scratch() {
    local resolved
    resolved=$(realpath -m -- "$1")
    case "$resolved" in
        "$TRELLIS_OWN_ROOT"|"$TRELLIS_OWN_ROOT"/*) ;;
        *) trellis_die "Path outside owned scratch subtree: $1" ;;
    esac
}
trellis_require_own_source() {
    TRELLIS_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
    trellis_require_scratch "$TRELLIS_SOURCE"
    cd -- "$TRELLIS_SOURCE"
}

# Writable destinations and executable payloads must stay in this source checkout.
# Shared fixtures outside it may be read explicitly by a checked payload.
trellis_require_owned() {
    local resolved
    [[ -n "${TRELLIS_SOURCE:-}" ]] || trellis_die "Source scope is not initialized"
    resolved=$(realpath -m -- "$1")
    case "$resolved" in
        "$TRELLIS_SOURCE"|"$TRELLIS_SOURCE"/*) ;;
        *) trellis_die "Path outside owned source subtree: $1" ;;
    esac
}
