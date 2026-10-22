date -Is
grep 'strix-trellis-fused-both-confirm' "$HOME/gpu-site/.gpu-turn/log" | tail -12 || true
cat "$HOME/gpu-site/.gpu-turn/holder" 2>/dev/null || true
ps -eo pid,ppid,ni,etime,args | grep -E '[g]pu-turn.*strix-trellis-fused-both-confirm|[p]robe_confirm|[r]un_confirm.sh|[t]imeout.*180s.*fused-both-confirm' || true
find runs controller -maxdepth 2 -type f -name '*started*' -o -name '*closed*' -o -name '*exit*' 2>/dev/null || true
for phase in search confirm; do
    if [[ -f runs/$phase/probe.log ]];then tail -6 runs/$phase/probe.log;cat runs/$phase/probe.stderr;fi
done
