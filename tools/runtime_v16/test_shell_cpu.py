"""Run the canonical full shell with CPU fakes and reject the historical cwd mutation."""
from pathlib import Path
import os, subprocess, tempfile, sys, json
repo=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(dir=out) as tmp:
    home=Path(tmp)/'home';root=home/'gpu-site/scratch-trellis';runtime=root/'runtime-v16';engine=root/'engine-v16'
    (runtime/'tools').mkdir(parents=True);(runtime/'build-hip').mkdir();(engine/'build/engine').mkdir(parents=True)
    (runtime/'tools/scratch_paths.sh').write_text((repo/'tools/scratch_paths.sh').read_text())
    runner=(repo/'tools/run_runtime_v16.sh').read_text()
    (runtime/'marker').write_text('parent');(engine/'marker').write_text('engine')
    import hashlib
    for directory in (runtime,engine):
        (directory/'frozen.sha256').write_text(hashlib.sha256((directory/'marker').read_bytes()).hexdigest()+'  marker\n')
    fixtures=root/'runtime-fixtures';fixtures.mkdir()
    for i in range(80):(fixtures/f'{i}.stgpu').touch()
    for name in ('gpu-dense-prefill-tests','gpu-numerical-tests','prefill-long/fixtures','k2-core/data'):
        directory=root/name;directory.mkdir(parents=True);(directory/'fixture.stgpu').touch()
    fake='''#!/usr/bin/env bash
set -euo pipefail
if [[ "$0" == *head_crosscheck ]]; then
 mkdir -p "$2";for r in {1..8};do touch "$2/r$r-a.f16" "$2/r$r-y.f32";done
fi
if [[ "$0" == *dense-example ]];then echo reached >> "$HOME/engine-reached";touch "$3";fi
echo CPU_FAKE
'''
    for version in ('runtime-v16','runtime-v14b'):
        for name in ('strix_trellis_gpu_runtime','strix_trellis_api_linear_bench','strix_trellis_head_linear_bench','strix_trellis_graph_linear_bench','head_crosscheck'):
            p=root/version/'build-hip'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(fake);p.chmod(0o755)
    p=engine/'build/engine/strix-trellis-dense-example';p.write_text(fake);p.chmod(0o755)
    fakebin=Path(tmp)/'bin';fakebin.mkdir();p=fakebin/'python3';p.write_text('#!/bin/sh\necho CPU_FAKE_VERIFIER\n');p.chmod(0o755)
    env=dict(os.environ,ROOT_GPU_LEASE='1',HOME=str(home),PATH=str(fakebin)+':'+os.environ['PATH'])
    results={}
    for mutation in (False,True):
        text=runner
        if mutation:
            assert '"$TRELLIS_SOURCE/frozen.sha256"' in text
            needle='(cd "$root/engine-v16" && sha256sum -c --quiet frozen.sha256'
            assert text.count(needle)==1
            text=text.replace(needle,'(cd "$root/engine-v16" && sha256sum -c --quiet qualification-v2/frozen.sha256 && sha256sum -c --quiet frozen.sha256')
        p=runtime/'tools/run_runtime_v16.sh';p.write_text(text)
        label='mutated' if mutation else 'correct'
        with (out/f'{label}.log').open('w') as log:
            run=subprocess.run(['bash',str(p),'--payload',label],env=env,stdout=log,stderr=subprocess.STDOUT)
        results[label]=run.returncode
        assert (run.returncode!=0)==mutation
        assert (runtime/f'runs/{label}/passed.txt').exists()!=mutation
        assert (runtime/f'runs/{label}/failed.txt').exists()==mutation
    assert (home/'engine-reached').read_text().splitlines()==['reached','reached']
(out/'proof.json').write_text(json.dumps(dict(state='CPU_SHELL_SCOPE_PASS_NO_HIP',results=results,engine_cases_reached=2,historical_mutation_rejected=True),indent=2)+'\n')
print('PASS CPU shell scope: two engine calls with correct cwd; historical mutation rejected')
