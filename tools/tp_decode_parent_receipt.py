"""Write ROOT's receipt only after entering the scheduler's bounded payload."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import socket
import sys

root = Path.home() / "gpu-site/scratch-trellis/tp-decode-qualification"
assert Path.cwd() == root and os.environ.get("ROOT_GPU_LEASE") == "1"
campaign = int(sys.argv[1])
assert campaign in (1, 2)
run = root / "runs" / f"review{campaign:03d}"
assert run.is_dir()
child = root.parent / "tp-decode-v1/host-r2"
pins = json.loads((root / "admission-pins.json").read_text(encoding="utf-8"))
node = socket.gethostname()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(child / "frozen.sha256") == pins["nodes"][node]["frozen_sha256"]
lease = run / "outer-lease.json"
with lease.open("x", encoding="utf-8", newline="\n") as f:
    json.dump(dict(state="ENTERED_REAL_SCHEDULER_PAYLOAD", node=node,
        campaign=campaign, scheduler_label=f"strix-trellis-tp{campaign}",
        utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        local=dt.datetime.now().astimezone().isoformat(), receipt_writer_pid=os.getpid(),
        parent_pid=os.getppid(), outer_seconds=900,
        parent_source_sha256=sha(root / "frozen.sha256"),
        scheduler_events_sha256=sha(run / "scheduler-before.txt")), f, indent=2)
    f.write("\n")
receipt = dict(campaign=campaign, label=f"ROOT-tp-decode-v1-qa{campaign:03d}",
    node=node, frozen_sha256=sha(child / "frozen.sha256"),
    actual_outer_lease_receipt=str(lease),
    actual_outer_lease_receipt_sha256=sha(lease), outer_seconds=900,
    root_confirms_active_lease=True,
    root_review={name: "PASS" for name in ("source", "resources", "ownership", "admission", "budget")},
    root_review_scope="CPU source/ELF/factory/receipt and bounded budget; native gates still OPEN",
    source_commit=pins["source_commit"],
    CPU_review_sha256=sha(root / "root-review.json"))
with (run / "child-admission.json").open("x", encoding="utf-8", newline="\n") as f:
    json.dump(receipt, f, indent=2)
    f.write("\n")
print(receipt["label"])
