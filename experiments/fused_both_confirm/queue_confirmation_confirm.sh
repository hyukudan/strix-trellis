[[ -f controller/first-campaign-qualified.txt ]]
grep -q '^PASS ' controller/first-campaign-qualified.txt
[[ ! -e controller/confirm-attempt1.started ]]
date -Is > controller/confirm-attempt1.started
printf '%s\n' "$$" > controller/confirm-attempt1.pid
set +e
bash experiments/fused_both_confirm/run_confirm.sh confirm > controller/confirm-attempt1.queue.log 2>&1
result=$?
set -e
printf '%s\n' "$result" > controller/confirm-attempt1.exit.txt
date -Is > controller/confirm-attempt1.closed.txt
cat controller/confirm-attempt1.queue.log
echo "CONTROLLER_EXIT=$result"
