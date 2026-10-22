attempt=${1:-attempt1}
[[ $attempt =~ ^attempt[1-9][0-9]*$ ]]
[[ ! -e controller/search-$attempt.started ]]
mkdir -p controller
date -Is > "controller/search-$attempt.started"
printf '%s\n' "$$" > "controller/search-$attempt.pid"
set +e
bash experiments/fused_both_confirm/run_confirm.sh search > "controller/search-$attempt.queue.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "controller/search-$attempt.exit.txt"
date -Is > "controller/search-$attempt.closed.txt"
cat "controller/search-$attempt.queue.log"
echo "CONTROLLER_EXIT=$result"
