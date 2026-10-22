#!/usr/bin/env bash
# All benchmark execution must enter through this script, never CTest.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/wmma_decoder_paths.sh"
trellis_require_own_source

check_safety() {
    # Force host local time even if the caller exported TZ.
    local minute now remaining
    now=$(TZ=:/etc/localtime date +%s)
    minute=$(TZ=:/etc/localtime date -d "@$now" +%H:%M)
    minute=$((10#${minute%:*} * 60 + 10#${minute#*:}))
    (( minute < 1260 || minute >= 1410 )) || trellis_die "Blocked: 21:00-23:30 local time"
    # Reserve the whole requested budget before 21:00, including check overhead.
    if (( minute < 1260 )); then
        remaining=$(( $(TZ=:/etc/localtime date -d 'today 21:00' +%s) - now ))
        (( seconds + 15 < remaining )) || trellis_die "Budget could overlap 21:00"
    fi
    timeout --signal=TERM --kill-after=1s 10s "$maintenance" ||
        trellis_die "Maintenance status unsafe, unknown, or check timed out"
}

if [[ ${1:-} == --inside ]]; then
    shift
    # Internal invocation is passed exclusively as the queue payload.
    [[ $# -ge 3 ]] || trellis_die "Invalid internal invocation"
    seconds=$1; maintenance=$2; shift 2
    [[ "$seconds" =~ ^[0-9]+$ && ${#seconds} -le 4 ]] || trellis_die "Invalid timeout"
    (( seconds == 180 )) || trellis_die "Invalid timeout"
    trellis_require_owned "$maintenance"
    [[ -x "$maintenance" ]] || trellis_die "Missing maintenance checker"
    check_safety
    exec timeout --signal=TERM --kill-after=5s "${seconds}s" "$@"
fi

seconds=180
maintenance=
queue_args=(strix-trellis)
while (( $# )); do
    case "$1" in
        --seconds|--maintenance-check|--queue-arg)
            (( $# >= 2 )) || trellis_die "Missing value for $1"
            case "$1" in
                --seconds) seconds=$2 ;;
                --maintenance-check) maintenance=$2 ;;
                --queue-arg) queue_args=("$2") ;;
            esac
            shift 2 ;;
        --) shift; break ;;
        *) trellis_die "Usage: $0 --maintenance-check PATH [--seconds 1..1180] [--queue-arg ARG]... -- COMMAND [ARG]..." ;;
    esac
done
[[ "$seconds" =~ ^[1-9][0-9]*$ && ${#seconds} -le 4 ]] || trellis_die "Invalid timeout"
(( seconds == 180 )) || trellis_die "This experiment requires a 180-second payload"
[[ -n "$maintenance" && $# -gt 0 ]] || trellis_die "Maintenance checker and command are required"
maintenance=$(realpath -e -- "$maintenance")
trellis_require_owned "$maintenance"
[[ -x "$maintenance" ]] || trellis_die "Maintenance checker must be executable"
# The benchmark executable must also live in this worktree's scratch subtree.
command_path=$(realpath -e -- "$1")
trellis_require_owned "$command_path"
[[ -x "$command_path" ]] || trellis_die "Command must be executable"
shift
queue="$HOME/gpu-site/gpu-turn-any.sh"
[[ -x "$queue" ]] || trellis_die "Approved queue wrapper is unavailable"
# Queue flags are operator-supplied; the wrapper must execute the appended argv
# synchronously inside its acquired turn. No guessed scheduler CLI is used.
check_safety
self="$TRELLIS_SOURCE/tools/run_wmma_packed_turn.sh"
# A bounded wait plus turn. TERM grace is included in the 1200-second ceiling.
timeout --signal=TERM --kill-after=5s 1195s "$queue" "${queue_args[@]}" \
    bash "$self" --inside "$seconds" "$maintenance" "$command_path" "$@"
