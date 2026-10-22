#!/usr/bin/env bash
# No real scheduler, remote host, maintenance service or GPU is accessed.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
test_root=$(mktemp -d)
trap 'rm -rf -- "$test_root"' EXIT
export HOME="$test_root/home"
own="$HOME/gpu-site/scratch-trellis/tooling"
mkdir -p "$own/tools" "$test_root/bin"
cp "$source_dir/tools/"{scratch_paths,run_gpu_turn,build_remote}.sh "$own/tools/"
export TEST_LOG="$test_root/log"
export TEST_TIME=12:00 TEST_REMAINING=32400 TEST_QUEUE_TIME=12:00
export TEST_MAINTENANCE=0 TEST_INSIDE_MAINTENANCE=0
cat > "$test_root/bin/date" <<'EOF'
#!/usr/bin/env bash
case "${*: -1}" in
    +%H:%M) echo "$TEST_TIME" ;;
    +%s)
        if [[ "$*" == *'today 21:00'* ]]; then
            echo "$((100000 + TEST_REMAINING))"
        else
            echo 100000
        fi ;;
    *) exit 1 ;;
esac
EOF
cat > "$HOME/gpu-site/gpu-turn-any.sh" <<'EOF'
#!/usr/bin/env bash
echo queue >> "$TEST_LOG"
[[ $# -ge 2 ]] || exit 1
shift # The production queue consumes its label before executing the payload.
export TEST_TIME="$TEST_QUEUE_TIME"
export TEST_MAINTENANCE="$TEST_INSIDE_MAINTENANCE"
exec "$@"
EOF
cat > "$own/maintenance" <<'EOF'
#!/usr/bin/env bash
echo maintenance >> "$TEST_LOG"
exit "$TEST_MAINTENANCE"
EOF
cat > "$own/benchmark" <<'EOF'
#!/usr/bin/env bash
echo payload >> "$TEST_LOG"
if [[ "${1:-}" == --sleep ]]; then sleep 10; exit 0; fi
[[ "${1:-}" == 'argument with spaces' ]]
EOF
chmod +x "$test_root/bin/date" "$HOME/gpu-site/gpu-turn-any.sh" \
    "$own/maintenance" "$own/benchmark"
export PATH="$test_root/bin:$PATH"
run() {
    bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
        --seconds 10 -- "$own/benchmark" 'argument with spaces'
}
reject() {
    if "$@" > "$test_root/output" 2>&1; then
        echo "Expected rejection: $*" >&2; exit 1
    fi
}
no_queue() { [[ ! -s "$TEST_LOG" ]]; }
no_payload() { ! grep -q '^payload$' "$TEST_LOG"; }
reset_log() { : > "$TEST_LOG"; }
reset_log
run
[[ "$(cat "$TEST_LOG")" == $'maintenance\nqueue\nmaintenance\npayload' ]]
for time in 21:00 22:15 23:29; do
    reset_log; export TEST_TIME="$time"
    reject run; no_queue
done
export TEST_TIME=23:30 TEST_QUEUE_TIME=23:30
reset_log; run
export TEST_TIME=12:00 TEST_QUEUE_TIME=12:00 TEST_REMAINING=20
reset_log; reject run; no_queue
export TEST_REMAINING=32400 TEST_MAINTENANCE=1
reset_log; reject run
[[ "$(cat "$TEST_LOG")" == maintenance ]]
export TEST_MAINTENANCE=0 TEST_QUEUE_TIME=21:00
reset_log; reject run; no_payload
[[ "$(cat "$TEST_LOG")" == $'maintenance\nqueue' ]]
export TEST_QUEUE_TIME=12:00 TEST_INSIDE_MAINTENANCE=1
reset_log; reject run; no_payload
[[ "$(cat "$TEST_LOG")" == $'maintenance\nqueue\nmaintenance' ]]
export TEST_INSIDE_MAINTENANCE=0
# Waiting may last longer than a GPU turn; its timeout must not replace the
# inside GPU timeout or bypass the second time/maintenance checks.
export REAL_TIMEOUT=$(command -v timeout)
export TEST_TIMEOUT_LOG="$test_root/timeouts"
cat > "$test_root/bin/timeout" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_TIMEOUT_LOG"
exec "$REAL_TIMEOUT" "$@"
EOF
chmod +x "$test_root/bin/timeout"
reset_log
bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
    --seconds 10 --controller-seconds 3600 -- "$own/benchmark" 'argument with spaces'
[[ "$(cat "$TEST_LOG")" == $'maintenance\nqueue\nmaintenance\npayload' ]]
grep -Fq -- "--kill-after=5s 3600s $HOME/gpu-site/gpu-turn-any.sh" "$TEST_TIMEOUT_LOG"
grep -Fq -- "--kill-after=5s 10s $own/benchmark" "$TEST_TIMEOUT_LOG"
export TEST_QUEUE_TIME=21:00
reset_log
reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
    --seconds 10 --controller-seconds 3600 -- "$own/benchmark" 'argument with spaces'
no_payload
export TEST_QUEUE_TIME=12:00 TEST_INSIDE_MAINTENANCE=1
reset_log
reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
    --seconds 10 --controller-seconds 3600 -- "$own/benchmark" 'argument with spaces'
no_payload
export TEST_INSIDE_MAINTENANCE=0
reset_log
for invalid in 0 15 7201 99999999999999999999; do
    reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
        --seconds 10 --controller-seconds "$invalid" -- "$own/benchmark"
    no_queue
done
reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
    --seconds 1181 --controller-seconds 7200 -- "$own/benchmark"
no_queue
reset_log
reject bash "$own/tools/run_gpu_turn.sh" --seconds 1201 \
    --maintenance-check "$own/maintenance" -- "$own/benchmark"
no_queue
reject bash "$own/tools/run_gpu_turn.sh" --seconds 0 \
    --maintenance-check "$own/maintenance" -- "$own/benchmark"
reject bash "$own/tools/run_gpu_turn.sh" --seconds 1181 \
    --maintenance-check "$own/maintenance" -- "$own/benchmark"
reject bash "$own/tools/run_gpu_turn.sh" --seconds 99999999999999999999 \
    --maintenance-check "$own/maintenance" -- "$own/benchmark"
reject bash "$own/tools/run_gpu_turn.sh" -- "$own/benchmark"
reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" -- /bin/true
no_queue
ln -s "$test_root/bin/date" "$own/escape"
reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/escape" -- "$own/benchmark"
no_queue
reset_log
bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
    --queue-arg --fake-prefix --seconds 1 -- "$own/benchmark" 'argument with spaces'
[[ "$(cat "$TEST_LOG")" == $'maintenance\nqueue\nmaintenance\npayload' ]]
# Existing sibling payloads and build symlinks cannot escape the source scope.
sibling="$HOME/gpu-site/scratch-trellis/another-worktree"
mkdir -p "$sibling"
cp "$own/benchmark" "$sibling/benchmark"
reset_log
reject bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" -- "$sibling/benchmark"
no_queue
ln -s "$sibling" "$own/build-hip"
reject bash -c 'source "$1/tools/scratch_paths.sh"; trellis_require_own_source; trellis_require_owned "$TRELLIS_SOURCE/build-hip"' _ "$own"
reset_log
status=0
bash "$own/tools/run_gpu_turn.sh" --maintenance-check "$own/maintenance" \
    --seconds 1 -- "$own/benchmark" --sleep > "$test_root/output" 2>&1 || status=$?
[[ "$status" == 124 ]]
[[ "$(cat "$TEST_LOG")" == $'maintenance\nqueue\nmaintenance\npayload' ]]
# Source outside the reserved subtree must fail before even checking ROCm.
reject bash "$source_dir/tools/build_remote.sh"
grep -q 'outside owned scratch subtree' "$test_root/output"
if compgen -G "$own/.gpu-turn.*" > /dev/null; then
    echo 'Queue ticket leaked' >&2; exit 1
fi
echo 'Safety checks passed (isolated fake queue, no GPU).'
