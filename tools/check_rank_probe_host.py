#!/usr/bin/env python3
"""Check native probe HOST C++ syntax with a declaration-only HIP placeholder.

This does not compile HIP kernels, link any executable, initialize HIP, execute
a mock GPU, validate the real HIP ABI, or establish native correctness/speed.
Use a real HIP build and the GPU turn runner for those checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DECLARATIONS = """#pragma once
#include <cstddef>
typedef int hipError_t;
constexpr hipError_t hipSuccess=0;
using hipStream_t=void*;using hipEvent_t=void*;using hipGraph_t=void*;using hipGraphExec_t=void*;using hipGraphNode_t=void*;
enum hipMemcpyKind{hipMemcpyHostToDevice,hipMemcpyDeviceToHost};
enum hipGraphNodeType{hipGraphNodeTypeKernel};
enum hipStreamCaptureMode{hipStreamCaptureModeThreadLocal};
constexpr unsigned hipStreamNonBlocking=1;
struct hipDeviceProp_t{char name[256];char gcnArchName[256];int warpSize;int multiProcessorCount;};
const char* hipGetErrorString(hipError_t);
hipError_t hipSetDevice(int);hipError_t hipGetDeviceProperties(hipDeviceProp_t*,int);
hipError_t hipStreamCreateWithFlags(hipStream_t*,unsigned);hipError_t hipStreamSynchronize(hipStream_t);hipError_t hipStreamDestroy(hipStream_t);
hipError_t hipMalloc(void**,size_t);hipError_t hipFree(void*);hipError_t hipMemsetAsync(void*,int,size_t,hipStream_t);
hipError_t hipMemcpyAsync(void*,const void*,size_t,hipMemcpyKind,hipStream_t);
hipError_t hipGraphExecDestroy(hipGraphExec_t);hipError_t hipGraphDestroy(hipGraph_t);
hipError_t hipStreamBeginCapture(hipStream_t,hipStreamCaptureMode);hipError_t hipStreamEndCapture(hipStream_t,hipGraph_t*);
hipError_t hipGraphGetNodes(hipGraph_t,hipGraphNode_t*,size_t*);hipError_t hipGraphNodeGetType(hipGraphNode_t,hipGraphNodeType*);
hipError_t hipGraphInstantiate(hipGraphExec_t*,hipGraph_t,hipGraphNode_t*,char*,size_t);hipError_t hipGraphLaunch(hipGraphExec_t,hipStream_t);
hipError_t hipEventCreate(hipEvent_t*);hipError_t hipEventRecord(hipEvent_t,hipStream_t);hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipEventElapsedTime(float*,hipEvent_t,hipEvent_t);hipError_t hipEventDestroy(hipEvent_t);
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    lines = ["HOST C++ SYNTAX ONLY: declaration-only HIP placeholder; no HIP compilation, ABI validation, linking or GPU/mock execution.",
             "UTC: " + datetime.now(timezone.utc).isoformat(), "cwd: " + str(ROOT)]
    version = subprocess.run([args.compiler, "--version"], text=True, capture_output=True, check=True)
    lines.append("compiler: " + version.stdout.splitlines()[0])
    paths = ("include/strix_trellis.h", "tests/rank_fixture.hpp", "tests/rank_fused_probe.hip",
             "tools/check_rank_probe_host.py")
    for path in paths:
        lines.append(json.dumps({"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}))
    lines.append("placeholder_sha256: " + hashlib.sha256(DECLARATIONS.encode()).hexdigest())
    with tempfile.TemporaryDirectory(prefix="strix-rank-host-") as temporary:
        directory = Path(temporary)
        (directory / "hip").mkdir()
        (directory / "hip/hip_runtime.h").write_text(DECLARATIONS, encoding="utf-8")
        command = [args.compiler, "-x", "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                   "-I" + str(directory), "-Iinclude", "-Itests", "tests/rank_fused_probe.hip"]
        lines.append("command: " + shlex.join(command))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.stdout:
            lines.append("stdout:\n" + result.stdout.rstrip())
        if result.stderr:
            lines.append("stderr:\n" + result.stderr.rstrip())
        lines.append("exit_code: " + str(result.returncode))
        lines.append("PASS host syntax only" if result.returncode == 0 else "FAIL host syntax")
    text = "\n".join(lines) + "\n"
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text, encoding="utf-8")
    print(text, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
