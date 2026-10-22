"""V16 CPU host adapter from HIGH 63bb781/5012c48; no remote or HIP execution."""
from pathlib import Path
import argparse
import re
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def raw(path):return (ROOT/path).read_bytes().replace(b"\r\n",b"\n")

def host_test(build, mutation=None):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    build.mkdir(exist_ok=True)
    include = build / "fake_hip/hip"
    include.mkdir(parents=True, exist_ok=True)
    (include / "hip_runtime.h").write_text('#define STRIX_API_FAKE_HIP\n#include "' +
        (ROOT / "tests/test_api_contract.cpp").as_posix() + '"\n#undef STRIX_API_FAKE_HIP\n')
    # Reuse only the existing fake runtime and probe adapters. The real API is
    # included there. Replace the old mocked backend with its actual host body.
    prefix = raw("tests/test_api_contract.cpp").decode().split("#else\n", 1)[1]
    prefix = prefix[:prefix.index("size_t linear_workspace_bytes(const DeviceWeights&, int rows)")] + "}\n"
    backend = raw("src/backend.hip").decode()
    backend = re.sub(r'^#include.*\n', '', backend, flags=re.M)
    backend, n = re.subn(r'input_transform<<<.*?>>>\(', 'cpu_input(', backend)
    assert n == 1
    backend, n = re.subn(r'output_transform<<<.*?>>>\(', 'cpu_output(', backend)
    assert n == 1
    launcher = raw("src/gemv_fused_both.hip").decode()
    launcher = launcher[launcher.index("hipError_t launch_fused_both_trusted"):]
    launcher, n = re.subn(r'fused_both::both<5,fused_both::Mode::Direct><<<.*?>>>\(', 'cpu_direct(layout,stream,', launcher)
    assert n == 1
    prelude = r'''
#include <string>
#include <algorithm>
#include "../src/launch_policy.hpp"
#include "../src/gemv_policy.hpp"
static std::vector<std::string> cpu_nodes;
static std::string cpu_fail;
static int cpu_last=0;
static int enqueue(const char* name) {
    cpu_nodes.emplace_back(name); return cpu_fail==name?99:0;
}
static int hipGetLastError() { int e=cpu_last; cpu_last=0; return e; }
namespace strix_trellis {
template<class... T> void cpu_input(T...) { cpu_last=enqueue("input"); }
template<class... T> void cpu_output(T...) { cpu_last=enqueue("output"); }
template<class... T> void cpu_direct(FusedBothLayout l,T...) {
    assert(l.blocks==40 && l.static_bytes==2048 && (l.dynamic_bytes==5120 || l.dynamic_bytes==16384));
    cpu_last=enqueue("direct");
}
template<class... T> int launch_gemv_fused_input(T...) { return enqueue("fused-input"); }
template<class... T> int launch_gemv_lds_core(T...) { return enqueue("lds"); }
template<class... T> int launch_wmma_core(T...) { return enqueue("wmma"); }
template<class... T> int launch_fused_output_trusted(const DeviceWeights& w,T...) {
    int groups=fused_output_groups(w.input,w.output,w.bits,w.codebook,8,4,false,w.fused_output_capable,true);
    assert(groups); return enqueue(groups==1?"old-direct-g1":"old-direct");
}
}
'''
    source = prefix + prelude + backend + "\nnamespace strix_trellis {\n" + launcher
    source += '\n' + raw("tools/runtime_v16/high_checks.cpp").decode()
    if mutation == "fallback-three-nodes":
        assert 'expect_nodes({"fused-input","output"})' in source
        source = source.replace('expect_nodes({"fused-input","output"})', 'expect_nodes({"input","fused-input","output"})')
    elif mutation == "couple-flags":
        source = source.replace('w.fused_both_capable,w.trellis', 'w.fused_output_capable,w.trellis')
    elif mutation == "post-enqueue-fallback":
        start = "return launch_fused_both_trusted(w,input,output,rows,prefer_wmma,stream,alpha,beta);"
        assert start in source
        source = source.replace(start, "(void)launch_fused_both_trusted(w,input,output,rows,prefer_wmma,stream,alpha,beta);")
    label = mutation or "host"
    path = build / (label + ".cpp")
    path.write_text(source)
    exe = build / (label + ".cpu")
    command = ["g++", "-std=c++17", "-O0", "-g", "-UNDEBUG", "-Wall", "-Wextra", "-Werror", "-pedantic",
               "-I" + str(build / "fake_hip"), "-I" + str(ROOT / "include"), str(path), "-o", str(exe)]
    print("CPU_COMPILE " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    result = subprocess.run([str(exe)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="")
    if mutation:
        assert result.returncode != 0, "Counterexample was not detected"
        print("PASS rejected counterexample=" + mutation)
    else:
        assert result.returncode == 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",action="store_true")
    parser.add_argument("--counterexamples",action="store_true")
    args=parser.parse_args()
    build=ROOT/"build-v16-review-cpu"
    if args.host:host_test(build)
    if args.counterexamples:
        for mutation in ("fallback-three-nodes","couple-flags","post-enqueue-fallback"):
            host_test(build,mutation)
