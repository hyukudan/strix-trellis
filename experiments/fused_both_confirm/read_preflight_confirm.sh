for file in build/host-rebuild.log build/host-harness.json build/host-guard.json build/cpu-normative.txt; do
    echo "$file"
    tail -8 "$file"
done
