python3 experiments/fused_both_confirm/host_harness_check.py > build/host-harness.json
python3 experiments/fused_both_confirm/identity_confirm.py > build/identity.json
python3 experiments/fused_both_confirm/cpu_validate_confirm.py > build/cpu-validation.json
bash experiments/fused_both_confirm/incident_receipt_confirm.sh
bash experiments/fused_both_confirm/run_confirm.sh --freeze
sha256sum -c build/frozen.sha256 > build/prequeue-frozen.txt
sha256sum -c build/fixtures.sha256 > build/prequeue-fixtures.txt
sha256sum -c build/old-preservation.sha256 > build/old-preservation-before-queue.txt
date -Is > build/prequeue.date.txt
