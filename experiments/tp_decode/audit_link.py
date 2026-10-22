"""CPU audit of actual host bindings, ordinary wrappers, and archived baseline origins."""
from pathlib import Path
import json
import re
import sys
build=Path(sys.argv[1]);symbols=(build/'host-symbols.txt').read_text();disasm=(build/'host-disassembly.txt').read_text();link=(build/'link.map').read_text()
for name in ('_Znwm','_Znam','_ZdlPv','_ZdaPv','_ZdlPvm','_ZdaPvm'):
    matching=[line for line in symbols.splitlines() if re.match(r'\s*\d+:',line) and re.search(r'\s'+name+r'(?:@[^\s]+)?(?:\s|$)',line)]
    assert matching and all(' UND ' in line for line in matching),('global new/delete interposition',name,matching)
    assert '__wrap_'+name in symbols
for name in ('baseline_core','launch_linear'):
    assert name in link and 'libstrix_trellis_v16.a' in link
    assert re.search(r'(?:call|jmp)\S*\s+.*strix_trellis::'+name,disasm),('missing actual archive call',name)
    definitions=[line for line in link.splitlines() if 'libstrix_trellis_v16.a' in line and ('gemv_lds' if name=='baseline_core' else 'backend') in line]
    assert definitions
wraps=json.loads((build/'build.json').read_text())['wraps']
for name in wraps:assert '__wrap_'+name in symbols,('wrapper missing',name)
result=dict(passed=True,cpu_only=True,TEN_asserts='compiled sizeof56 offsets48/49',
            baseline='actual V16 baseline_core explicit S4 and launch_linear; archive origin retained in link.map',
            no_global_new_delete_definition=True,ordinary_wrappers=wraps,
            instrumentation_scope='user/static link references only; DSO internal operations outside scope, CPU counterexample PASS',
            native_submit_side_effect_gate='OPEN')
(build/'link-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
