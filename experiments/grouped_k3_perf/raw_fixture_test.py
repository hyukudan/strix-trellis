"""SYNTHETIC CPU log format fixture only. Never a real campaign or proposal."""
from pathlib import Path
import tempfile,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from provenance import raw_pass
from provenance_test import rejects

def main():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-CPU-raw-') as tmp:
        log=Path(tmp)/'SYNTHETIC.log'
        with log.open('w') as f:
            f.write('SYNTHETIC CPU FORMAT FIXTURE; NO REAL PERFORMANCE PROPOSAL\n')
            for i in (5120,2304):
                for g in (4,8,16,32):
                    for mode in ('direct','graph'):
                        for variant in ('V13','new'):
                            for sweep in range(11):
                                for clone in range(64//g):f.write('SAMPLE '+json.dumps(dict(input=i,G=g,mode=mode,variant=variant,sweep=sweep,clone=clone,event_ms=1,wall_ms=2))+'\n')
            for i in (5120,2304):
                f.write('ledger explicit_gpu_bytes=306905088 packed_resident=283115520 host_limit=2147483648 contexts_observed=1\n')
                for g in (4,8,16,32):
                    for v in (0,1):
                        for c in range(64//g):
                            for index in range(3 if v else 3*g):f.write(f'GRAPH clone_start={c*g} variant={v} index={index} func=0x100 grid=1,1,1 block=128 shared=0\n')
                    for _ in range(27):
                        for j in range(64):
                            for split in range(16 if i==5120 else 4):f.write(f'CHECK id={j} split={split} stage=P64 maxabs=0 nrms=0 maxref=1 worst=0 enforce=1\n')
                            for stage,enforce in (('Yordered',1),('YoriginalFP64',0)):f.write(f'CHECK id={j} split=-1 stage={stage} maxabs=0 nrms=0 maxref=1 worst=0 enforce={enforce}\n')
                            f.write(f'SUMABS id={j} ratio=0 empirical_only=1\n')
                        f.write(f'PASS_VERIFY input={i} G={g} coverage=64 ownership=PASS guards=PASS immutable=PASS RSS=PASS ledger=PASS A_P_Y=PASS\n')
            f.write('PASS physical performance campaign all fixed cells; contexts=2 calls=14\n')
            f.write('attribute_scope=observed_link_boundary_user_and_static_library_calls_not_internal_driver_queries logical_creations=2 contexts_with_handle=2 contexts_admitted=2 calls=14 stored=14 overflow=0 failed=0 first_error=0 trace_bytes=12672\n')
            for j in range(14):f.write(f'attribute_event={j} creation={j//7+1} context=0x{j//7+1} function=0x{j%7+1} phase={"creation" if j%7<4 else "math"} result=0 valid=1 source_order_binding=SYNTHETIC maxthreads=128 local=0 static=0 const=0 regs=1 binary=115 ptx=115\n')
            for phase in ('prepare','submit','capture','replay','reuse','cleanup'):f.write(f'attribute_zero_region phase={phase} regions=1 delta=0\n')
        text=raw_pass(log)
        for before,after in [('nrms=0','nrms=0.01'),('variant=0 index=0','variant=0 index=1'),('coverage=64','coverage=63'),('delta=0','delta=1'),('SYNTHETIC CPU','STOP SYNTHETIC CPU')]:
            log.write_text(text.replace(before,after,1));rejects(lambda:raw_pass(log))
        print('PASS SYNTHETIC CPU raw format positive and 5 negative mutations; no real performance proposal')
if __name__=='__main__':main()
