"""Exercise the actual host stimulus and CPU checkpoints before freezing/queueing."""
from pathlib import Path
import hashlib
import json
import subprocess

root=Path.cwd()
source=(root/'experiments/fused_both_confirm/probe_confirm.hip').read_text()
old=(root/'experiments/fused_both/probe.hip').read_text()
assert 'weights,ys,bps,cps,workspaces' in source
assert 'auto& dy=*ys[c]' in source
assert 'request(w,dx,dy,&cp' in source
assert 'auto got=ys[c]->read()' in source
assert 'Every clone last measured full Y' in source
assert 'all_clones_last_Y_exact=1 full_cpu=1' in source
assert 'if(sweep==10)last_measured[path][c]=got' in source
assert 'Retained every mode/clone last measured Y' in source
assert 'LAST_ALL_MODES name=' in source
assert 'for(size_t c=0;c<clones;++c){auto got=ys[c]->read()' not in old
assert source.count('0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,0x0400,0x8400')==1
assert 'u[k]=scales[k%12]' in source and 'v[c]=scales[c%12]' in source
assert 'x[4]=.2001299113035202f;u[4]=halfbits(.02642822265625f)' in source
assert '__global__ void' not in source.replace('__global__ void input_transform','').replace('__global__ void output_transform','')

def function(name):
    # Locate the declaration directly, independent of the preceding function length.
    import re
    begin=list(re.finditer(r'^static [^\n]+\b'+name+r'\(',source,re.M))[0].start()
    brace=source.index('{',begin); depth=1; end=brace+1
    while depth:
        depth += (source[end]=='{')-(source[end]=='}');end+=1
    return source[begin:end]+'\n'

names=['require','half','halfbits','cpu_h','weight','cpu_core','cpu_partial','cpu_output','cpu_input']
program='#include <fstream>\n#include <algorithm>\n#include <cmath>\n#include <cstring>\n#include <stdexcept>\n#include <vector>\n#include <cstdint>\n#include <iostream>\n'
program+=''.join(function(name) for name in names)
stimulus=source[source.index('        float patterns[]='):source.index('        Buffer dp(')]
program+='''int main(){size_t cases=0;
for(int bits:{5,6})for(int i:{128,256,384,512,640,1280,2304,5120,8192})for(int n:{128,256}){
std::vector<uint32_t> p(size_t(i)*n*bits/32);uint32_t rng=123+i+n+bits;
for(auto& x:p){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=rng;}
std::vector<float>x(i),b(n),old(n);std::vector<uint16_t>u(i),v(n);
'''+stimulus+'''
for(float z:cpu_p)require(std::isfinite(z),"Finite independent CPU partial");
for(bool bias:{false,true})for(float alpha:{0.f,-0.f,1.f,-.75f,.5f})for(float beta:{0.f,-0.f,1.f,-.75f,.5f}){
auto y=cpu_output(core,v,bias?b:std::vector<float>{},old,alpha,beta,n);
for(float z:y)require(std::isfinite(z),"Finite all CPU Y");++cases;}
for(uint16_t bits:scales){require(std::find(u.begin(),u.end(),bits)!=u.end(),"Missing suh bits");require(std::find(v.begin(),v.end(),bits)!=v.end(),"Missing svh bits");}
}
for(const char* name:{"q_b","wo_b","inverse","shared","head"}){
std::string folder=std::string("fixtures/")+name;bool head=std::string(name)=="head";
int i=std::string(name)=="q_b"?1280:std::string(name)=="wo_b"?8192:std::string(name)=="inverse"?2304:5120;
int n=std::string(name)=="q_b"?32768:std::string(name)=="wo_b"?5120:std::string(name)=="inverse"?5120:std::string(name)=="shared"?2304:129280;
std::vector<float>x(i),h,expected_h(i);std::vector<uint16_t>u(i),expected_a(i);
std::ifstream fx(folder+(head?"/input.f32":"/x.bin"),std::ios::binary);require(bool(fx.read((char*)x.data(),i*4)),"Read full X");
std::ifstream fu(folder+(head?"/head.blob":"/u.bin"),std::ios::binary);if(head)fu.seekg(size_t(i)*n*6/8);require(bool(fu.read((char*)u.data(),i*2)),"Read full suh");
auto a=cpu_input(x,u,h);
std::ifstream fa(folder+"/cpu-a.f16",std::ios::binary);require(bool(fa.read((char*)expected_a.data(),i*2)),"Read full CPU A");
std::ifstream fh(folder+"/cpu-h.f32",std::ios::binary);require(bool(fh.read((char*)expected_h.data(),i*4)),"Read full CPU H");
require(!std::memcmp(a.data(),expected_a.data(),i*2)&&!std::memcmp(h.data(),expected_h.data(),i*4),"Independent complete CPU A/H bytes");
std::cout<<"HOST_LARGE_AH "<<name<<" I="<<i<<" exact_bytes=1"<<std::endl;
}
std::cout<<"PASS actual_host_cpu epilogues="<<cases<<" all_A_H_P_Y_finite=1 explicit_scale_bits=12 tie=7530 clone_isolation_source=PASS\\n";
}'''
folder=root/'build/host-harness';folder.mkdir(parents=True,exist_ok=True)
(folder/'cpu_confirm.cpp').write_text(program)
subprocess.run(['g++','-O3','-ffp-contract=off','-std=c++17',str(folder/'cpu_confirm.cpp'),'-o',str(folder/'cpu_confirm')],check=True)
result=subprocess.check_output([str(folder/'cpu_confirm')],text=True)
print(json.dumps(dict(probe_source_sha256=hashlib.sha256(source.encode()).hexdigest(),output=result,
                     scope='Compiled actual CPU/stimuli plus source isolation checks; no HIP runtime or GPU attributes')))
