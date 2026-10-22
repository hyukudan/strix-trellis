# Building and running
> **Type:** user documentation.

## CPU and local CI

Requirements: Python >=3.10, NumPy >=1.24, CMake >=3.24, a C11/C++17 compiler,
Bash and GNU coreutils on Linux. The default configuration does not load HIP.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ./python
bash tools/ci_cpu.sh
```

Local CI configures Release, builds and runs fifteen CTest tests on Linux/WSL:
GEMV/V14/local K2/dense TP/TP MoE policies, checkpoints and the A/B protocol,
the head parser, C/C++ headers, real V14b factories with a simulated GPU,
runner scopes, general and TP API contracts with mock HIP, and GPU-turn safety.
Native Windows registers policies, the parser, headers and A/B checkpoints:
nine tests. It then runs Python unittest.
External fixtures are skipped when unavailable; report those skips when
publishing a result. `STRIX_TRELLIS_FIXTURES` selects the real range-fixture
directory. Consult the `fetch_*` scripts and `tests/fixtures` manifests;
do not download entire shards for a small test.

The GitHub workflow uses this CPU path. On 20 and 21 October 2026, jobs could not
start because of an account billing/spending-limit issue; that does not
count as a test run. Local validation does not depend on Actions.

The [README](../README.md#build-with-hip) contains the CMake commands and
build options. [TP_RANK.md](TP_RANK.md) details the build and the three
mix-B/expert-TP probe modes, fixtures without network access, hashes and
GPU turns. `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP` and
`STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL` remain disabled by default.

## HIP on gfx1151

`tools/run_gpu_turn.sh --seconds N` limits the GPU payload to at most
1180 s. `--controller-seconds N` limits waiting plus that payload:
1195 s by default, extendable to 7200 s for long queues.
Extending the controller does not extend the GPU turn.
Site-specific quiet hours and maintenance are checked before queueing
and again within the turn. Do not repeat a request until the previous
request and any payload it started have been confirmed to have ended.

Use separate CPU and HIP build directories.
On test machines, use a dedicated checkout under a scratch directory.

```sh
cd "<checkout>"
bash tools/build_remote.sh
```

The script configures ROCm `/opt/rocm/core-10.1`, `gfx1151`, and caches and
temporary files within the checkout. It builds with `nice -n 10` and two jobs.
It does not execute GPU work. Alternatively, use the README's CMake commands.
It produces `libstrix_trellis.a`, target `strix_trellis::runtime`, and the
`strix_trellis_gpu_runtime` harness. `strix_trellis::api` supplies headers only.
GPU programs are never registered as CTest tests.

## GPU turns

All GPU execution goes through `tools/run_gpu_turn.sh`, which wraps
a site-specific GPU queue script (not part of this repository).
The GPU scripts in `tools/` expect this site layout:

| Path | Purpose |
|---|---|
| `$HOME/gpu-site/gpu-turn-any.sh <label> <command>` | Site queue script that grants one bounded GPU turn (not included). |
| `$HOME/gpu-site/.gpu-turn/maintenance` | If this file exists, no GPU work is queued. |
| `$HOME/gpu-site/scratch-trellis/` | Scratch tree; checkouts, builds and runs must live under it. |

The site queue selects a test machine. The payload, fixtures and libraries
must be available at the same paths on both test machines (two Strix Halo systems).
Check hashes before execution, and do not replace a binary while it is queued.

```sh
bash tools/run_gpu_turn.sh --seconds 180 \
  --maintenance-check "$PWD/tools/check_maintenance.sh" \
  --queue-arg strix-trellis-check -- \
  "$PWD/build-hip/strix_trellis_gpu_runtime" --route all /path/to/fixture.stgpu
```

The command and checker must be inside the canonical checkout;
symlinks to a sibling directory are rejected.
Shared fixtures may be read explicitly by the payload.
The checker checks site-specific maintenance status for that queue.
It uses the host's time, ignoring the caller's TZ.
It blocks site-specific quiet hours, reserves a margin before they begin,
and repeats the check within the turn.

The payload accepts 1–1180 seconds; waiting plus execution has an outer
timeout of 1195 seconds with five seconds of grace.
A long queue can consume the budget without execution.
A timeout is not a benchmark result.
`--inside` is an internal queue-payload detail and must not be invoked directly.

The helpers protect cooperative tools; they are not a sandbox against arbitrary
commands. Do not modify other users' files or processes.
For every published figure, retain the command, exact source/binary, test
machine, date, correctness results and log.
