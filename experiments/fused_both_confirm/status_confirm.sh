date -Is
for file in build-confirm.log prepare-confirm.log build/identity.json build/cpu-references.jsonl; do
    if [[ -f $file ]]; then echo "$file"; tail -8 "$file"; fi
done
ps -eo pid,ppid,ni,etime,args | grep -E '[f]used-both-confirm|[p]robe_confirm' || true
