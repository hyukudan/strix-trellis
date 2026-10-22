# strix-trellis

Independent EXL3/QTIP implementation for AMD gfx1151. Work in progress.

## Development rules

- Reimplement algorithms independently; do not copy upstream implementation code.
- Code, commits and documentation in English.
- Commit author: hyukudan <19479239+hyukudan@users.noreply.github.com>. No co-author trailers.
- Benchmark claims require recorded commands, hardware, correctness and measured results. Mark unmeasured estimates explicitly.
- Do not compare performance with other inference engines or other vendors' hardware in public documentation. Format and paper attribution is required.
- Do not commit model weights or full model shards; fixtures use bounded byte ranges.
- Keep GPU runs bounded and go through `tools/run_gpu_turn.sh`; never run GPU benchmarks on a shared machine outside its queue.

