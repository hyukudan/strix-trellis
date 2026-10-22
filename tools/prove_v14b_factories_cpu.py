"""Compile actual host factories with mocked submissions; never compile GPU bodies."""
from pathlib import Path
import subprocess, sys, json

repo=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
def body(path,signature):
    text=(repo/path).read_text();start=text.index(signature);op=text.index('{',start);end=op+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]

packed=body('src/wmma_packed.hip','hipError_t launch_wmma_packed(')
start=packed.index('    dim3 grid(')
packed=packed[:start]+'    ++packed_calls;return packed_status;\n}'
public=body('src/gemm.hip','hipError_t launch_wmma_core(')
k2=body('src/gemv_k2_multirow.hip','hipError_t launch_gemv_k2_multirow(')
k2=k2[:k2.index('    dim3 grid(')]+'    ++k2_calls;return 0;\n}'
(out/'hip').mkdir(exist_ok=True)
(out/'hip/hip_runtime.h').write_text('using hipError_t=int;using hipStream_t=void*;constexpr int hipErrorInvalidValue=1;\n')
source='''#include <cassert>
#include "backend.hpp"
#include "launch_policy.hpp"
#include "v14_core_policy.hpp"
namespace strix_trellis {
int packed_calls=0,baseline_calls=0,k2_calls=0,packed_status=7;
template<int M,bool X=false,int P=0> int launch_tile(const DeviceWeights&,const _Float16*,float*,int,int,void*) {
    assert(M==64 && X && P==2);++baseline_calls;return 9;
}
'''+packed+'\n'+public+'\n'+k2+'''
}
int main(){
using namespace strix_trellis;
static_assert(sizeof(DeviceWeights)==56 && offsetof(DeviceWeights,fused_output_capable)==48);
alignas(4) _Float16 a[4]{};alignas(4) float p[4]{};uint32_t t=0;
const int tuples[][4]={{1280,32768,5,1},{8192,5120,5,4},{5120,2304,5,8},{2304,5120,5,4},{4096,1024,5,16},{5120,1280,6,16}};
int accepted=0,removed=0;
for(int shape=0;shape<6;++shape)for(int slot=0;slot<3;++slot){
 if(shape==5 && slot!=2)continue;
 int m=64<<slot,s=tuples[shape][3]>>slot;if(shape==0)s=1;
 DeviceWeights w{&t,nullptr,nullptr,nullptr,tuples[shape][0],tuples[shape][1],tuples[shape][2],2};
 bool eligible=!(shape==2 && slot==2 || shape==4 && slot!=2);
 int pc=packed_calls,bc=baseline_calls;
 int result=launch_wmma_packed(w,a,p,m,s,nullptr);
 assert(result==(eligible?7:hipErrorInvalidValue));assert(packed_calls-pc==int(eligible));
 pc=packed_calls;result=launch_wmma_core(w,a,p,m,s,nullptr);
 assert(result==(eligible?7:9));assert(packed_calls-pc==int(eligible));assert(baseline_calls-bc==int(!eligible));
 accepted+=eligible;removed+=!eligible;
 for(int r=0;r<=513;++r)for(int split=0;split<=33;++split){
   bool e=wmma_packed_candidate(w.input,w.output,w.bits,2,r,split,wmma_tile_rows(w,r),
       (w.bits==5 && w.codebook==2)||medium_prefill_candidate(w,r),true,true,true);
   pc=packed_calls;int status=launch_wmma_packed(w,a,p,r,split,nullptr);
   assert(status==(e?7:hipErrorInvalidValue));assert(packed_calls-pc==int(e));
 }
 assert(launch_wmma_packed(w,a+1,p,m,s,nullptr)==hipErrorInvalidValue);
 assert(launch_wmma_packed(w,nullptr,p,m,s,nullptr)==hipErrorInvalidValue);
}
assert(accepted==13 && removed==3);
for(auto shape:{0,1})for(int r=0;r<=513;++r)for(int s=0;s<=33;++s)for(int delta:{-1,0,1}){
 DeviceWeights w{&t,nullptr,nullptr,nullptr,shape?1280:8192,shape?32768:5120,2,2};
 bool e=(r==2 || r==4 || r==8)&&s==4&&delta==0;int before=k2_calls;
 assert(launch_gemv_k2_multirow(w,a,p,r,r+delta,s,nullptr)==(e?0:hipErrorInvalidValue));
 assert(k2_calls-before==int(e));
 assert(launch_gemv_k2_multirow(w,a+1,p,r,r+delta,s,nullptr)==hipErrorInvalidValue);
}
}
'''
source=source.replace('#include <cassert>','#include <cassert>\n#include <initializer_list>')
(out/'factories.cpp').write_text(source)
command=['g++','-O2','-std=c++17','-UNDEBUG','-I'+str(out),'-I'+str(repo/'src'),str(out/'factories.cpp'),'-o',str(out/'factories_cpu')]
subprocess.run(command,check=True);subprocess.run([str(out/'factories_cpu')],check=True)
(out/'proof.json').write_text(json.dumps(dict(state='CPU_MOCK_ACTUAL_HOST_FACTORIES_PASS',eligible=13,removed=3,rows=[0,513],splits=[0,33],source_files=['src/gemm.hip','src/wmma_packed.hip','src/gemv_k2_multirow.hip'],command=command),indent=2)+'\n')
print('PASS actual host factories: 13 accepted, 3 trusted rejects and baseline fallthrough; K2/chunk/alignment')
