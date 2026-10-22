bash experiments/fused_both_confirm/run_confirm.sh --build > build-confirm.log 2>&1
bash experiments/fused_both_confirm/run_confirm.sh --fixtures > prepare-confirm.log 2>&1
python3 experiments/fused_both/host_check.py > build/host-guard.json
python3 -I -S -B experiments/fused_both/source_proof.py > build/source-proof.json
# Normative proof reads its immutable original tree and does not write there.
python3 -I -S -B "$HOME/gpu-site/scratch-trellis/fused-both-core/reports/fused-both-review-proof.py" > build/cpu-normative.txt
