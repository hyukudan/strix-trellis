"""Synthetic CPU parser tests only; never creates or claims physical run records."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('v14',Path(__file__).with_name('verify_runtime_v14.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=['SETUP rows=8']*20+[f'RESULT mode={mode} rows=8' for mode in ('direct','graph_individual','graph_rotation') for _ in range(20)]+['PASS complete_operator_graphs']
m.verify_graph_records(rows)
checks=1
for bad in [rows[:19]+rows[20:],rows[:20]+rows[21:],rows[:-1]+['PASS'],rows[:20]+['RESULT mode=direct rows=8']*21+['RESULT mode=graph_individual rows=8']*19+['RESULT mode=graph_rotation rows=8']*20+rows[-1:],[],['RESULT mode=direct']*20+rows[-1:]]:
    try:m.verify_graph_records(bad)
    except AssertionError:checks+=1
    else:raise AssertionError('Malformed synthetic graph records accepted')
m.verify_head_terminal(['PASS sampled_head_operator (sampled numerical coverage)']);checks+=1
for rows in [[],['PASS sampled_head_operator'],['PASS sampled_head_operator (sampled numerical coverage) trailing']]:
    try:m.verify_head_terminal(rows)
    except AssertionError:checks+=1
    else:raise AssertionError('Wrong synthetic head terminal accepted')
print(f'PASS CPU_SYNTHETIC_RECORD_PARSER checks={checks} graphs=20_SETUPS_60_RESULTS_20_PER_MODE native_execution=False')
