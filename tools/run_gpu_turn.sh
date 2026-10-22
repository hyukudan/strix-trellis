#!/usr/bin/env bash
# All benchmark execution must enter through this script, never CTest.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
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
    (( seconds >= 1 && seconds <= 1180 )) || trellis_die "Invalid timeout"
    trellis_require_owned "$maintenance"
    [[ -x "$maintenance" ]] || trellis_die "Missing maintenance checker"
    check_safety
    exec timeout --signal=TERM --kill-after=5s "${seconds}s" "$@"
fi

seconds=600
controller_seconds=1195
maintenance=
queue_args=(strix-trellis)
while (( $# )); do
    case "$1" in
        --seconds|--controller-seconds|--maintenance-check|--queue-arg)
            (( $# >= 2 )) || trellis_die "Missing value for $1"
            case "$1" in
                --seconds) seconds=$2 ;;
                --controller-seconds) controller_seconds=$2 ;;
                --maintenance-check) maintenance=$2 ;;
                --queue-arg) queue_args=("$2") ;;
            esac
            shift 2 ;;
        --) shift; break ;;
        *) trellis_die "Usage: $0 --maintenance-check PATH [--seconds 1..1180] [--controller-seconds N] [--queue-arg ARG]... -- COMMAND [ARG]..." ;;
    esac
done
[[ "$seconds" =~ ^[1-9][0-9]*$ && ${#seconds} -le 4 ]] || trellis_die "Invalid timeout"
(( seconds <= 1180 )) || trellis_die "Maximum payload is 1180 seconds (20s safety overhead)"
[[ "$controller_seconds" =~ ^[1-9][0-9]*$ && ${#controller_seconds} -le 4 ]] || trellis_die "Invalid controller timeout"
(( controller_seconds >= seconds + 15 && controller_seconds <= 7200 )) ||
    trellis_die "Controller timeout must cover payload plus 15 seconds and be at most 7200 seconds"
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
self="$TRELLIS_SOURCE/tools/run_gpu_turn.sh"
# The controller includes waiting time. The independent inside timeout above
# always limits GPU payloads to 1180 seconds, even with a longer queue wait.
timeout --signal=TERM --kill-after=5s "${controller_seconds}s" "$queue" "${queue_args[@]}" \
    bash "$self" --inside "$seconds" "$maintenance" "$command_path" "$@"
