#include "trellis_device.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#define HIP(call) do { auto status = (call); if (status != hipSuccess) { std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(status)); std::exit(1); } } while(0)
using namespace strix_trellis;

__global__ void contiguous_read(const uint4* data, uint32_t* result, size_t vectors) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t checksum = 0;
    for (; index < vectors; index += size_t(gridDim.x) * blockDim.x) {
        uint4 item = data[index];
        checksum ^= item.x ^ item.y ^ item.z ^ item.w;
    }
    for (int offset = 16; offset; offset /= 2) checksum ^= __shfl_down(checksum, offset, 32);
    if ((threadIdx.x & 31) == 0) result[(size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32] = checksum;
}

template<int B, int Mode, int Codebook>
__global__ void tile_scan(const uint32_t* data, float* checksum, size_t tiles) {
    size_t tile = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (tile >= tiles) return;
    WaveTile<B> packed(data + tile * B * 8);
    float total = 0;
    if constexpr (Mode == 0) {
        total = float(packed.lo & 65535u) + float(packed.lo >> 16);
        if constexpr (B > 4) total += float(packed.hi & 65535u) + float(packed.hi >> 16);
    } else if constexpr (Mode == 3) {
        constexpr int group = B <= 4 ? 8 : (B == 7 ? 2 : 4);
        #pragma unroll
        for (int base = 0; base < 8; base += group) {
            uint64_t window = packed.template group<group>((threadIdx.x & 31) * 8 + base);
            #pragma unroll
            for (int j = 0; j < group; ++j) {
                unsigned state = uint32_t(window >> ((group - 1 - j) * B)) & 65535u;
                total += codebook_value(state, Codebook);
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t state = packed.state((threadIdx.x & 31) * 8 + j);
            if constexpr (Mode == 1) total += float(state);
            else total += codebook_value(state, Codebook);
        }
    }
    for (int offset = 16; offset; offset /= 2) total += __shfl_down(total, offset, 32);
    if ((threadIdx.x & 31) == 0) checksum[tile] = total;
}

template<int B, int Mode>
void launch_scan(dim3 grid, dim3 block, const uint32_t* data, float* result, size_t tiles, int cb) {
    if (cb == 0) tile_scan<B,Mode,0><<<grid,block>>>(data,result,tiles);
    if (cb == 1) tile_scan<B,Mode,1><<<grid,block>>>(data,result,tiles);
    if (cb == 2) tile_scan<B,Mode,2><<<grid,block>>>(data,result,tiles);
}

template<int B>
__global__ void validate_states(const uint32_t* data, uint16_t* states, uint16_t* values, size_t tiles, int cb) {
    size_t tile = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (tile >= tiles) return;
    WaveTile<B> packed(data + tile * B * 8);
    for (int j = 0; j < 8; ++j) {
        int pos = (threadIdx.x & 31) * 8 + j;
        unsigned state = packed.state(pos);
        constexpr int group = B <= 4 ? 8 : (B == 7 ? 2 : 4);
        uint64_t window = packed.template group<group>((pos / group) * group);
        unsigned grouped = uint32_t(window >> ((group - 1 - pos % group) * B)) & 65535u;
        // A deliberately invalid state flags any optimized-extraction mismatch.
        states[tile * 256 + pos] = state == grouped ? state : (state ^ 1);
        values[tile * 256 + pos] = __half_as_ushort(__float2half_rn(codebook_value(state, cb)));
    }
}

#include "host_reference.hpp"

template<int B> void run(size_t target_bytes, int cb, int repeats) {
    const size_t tiles = target_bytes / (32 * B);
    const size_t bytes = tiles * 32 * B;
    std::vector<uint32_t> host(bytes / 4);
    uint32_t rng = 0x12345678;
    for (auto& word : host) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; word = rng; }
    uint32_t* data; float* result;
    HIP(hipMalloc(&data, bytes)); HIP(hipMalloc(&result, tiles * sizeof(float)));
    HIP(hipMemcpy(data, host.data(), bytes, hipMemcpyHostToDevice));
    const size_t check_tiles = std::min<size_t>(257, tiles);
    uint16_t *states, *values;
    HIP(hipMalloc(&states, check_tiles * 512)); HIP(hipMalloc(&values, check_tiles * 512));
    validate_states<B><<<(check_tiles + 7) / 8, 256>>>(data, states, values, check_tiles, cb);
    HIP(hipGetLastError());
    std::vector<uint16_t> got(check_tiles * 256), weights(got.size());
    HIP(hipMemcpy(got.data(), states, got.size() * 2, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(weights.data(), values, weights.size() * 2, hipMemcpyDeviceToHost));
    for (size_t tile = 0; tile < check_tiles; ++tile) for (int pos = 0; pos < 256; ++pos) {
        uint16_t expected = host_state(host.data() + tile * B * 8, B, pos);
        if (got[tile * 256 + pos] != expected || weights[tile * 256 + pos] != host_codebook(expected, cb)) {
            std::fprintf(stderr,"FAIL bits=%d cb=%d tile=%zu pos=%d state=%u/%u weight=%u/%u\n",B,cb,tile,pos,got[tile*256+pos],expected,weights[tile*256+pos],host_codebook(expected,cb)); std::exit(2);
        }
    }
    HIP(hipFree(states)); HIP(hipFree(values));
    hipEvent_t start, stop; HIP(hipEventCreate(&start)); HIP(hipEventCreate(&stop));
    for (int mode = 0; mode < 5; ++mode) {
        auto launch = [&]() {
            dim3 grid((tiles + 7) / 8), block(256);
            if (mode == 0) launch_scan<B,0>(grid,block,data,result,tiles,cb);
            if (mode == 1) launch_scan<B,1>(grid,block,data,result,tiles,cb);
            if (mode == 2) launch_scan<B,2>(grid,block,data,result,tiles,cb);
            if (mode == 3) launch_scan<B,3>(grid,block,data,result,tiles,cb);
            if (mode == 4) contiguous_read<<<512,256>>>((const uint4*)data,(uint32_t*)result,bytes/16);
            HIP(hipGetLastError());
        };
        launch(); HIP(hipDeviceSynchronize());
        std::vector<float> times;
        for(int repeat=0; repeat<repeats; ++repeat) {
            HIP(hipEventRecord(start)); launch(); HIP(hipEventRecord(stop)); HIP(hipEventSynchronize(stop));
            float ms; HIP(hipEventElapsedTime(&ms,start,stop)); times.push_back(ms);
        }
        std::sort(times.begin(),times.end()); float ms = times[times.size()/2];
        std::printf("{\"benchmark\":\"tile_scan\",\"bits\":%d,\"codebook\":%d,\"mode\":%d,\"bytes\":%zu,\"repeats\":%d,\"median_ms\":%.6f,\"packed_GBs\":%.3f,\"correctness\":\"bit_exact_states_and_fp16\"}\n",B,cb,mode,bytes,repeats,ms,double(bytes)/ms/1e6);
        std::fflush(stdout);
    }
    HIP(hipEventDestroy(start)); HIP(hipEventDestroy(stop)); HIP(hipFree(data)); HIP(hipFree(result));
}
static int check_fixture(const char* packed_path, const char* states_path, const char* values_path) {
    auto read = [](const char* path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open fixture");
        return std::vector<char>(std::istreambuf_iterator<char>(file), {});
    };
    auto input = read(packed_path), expected_states = read(states_path), expected_values = read(values_path);
    if (input.empty() || input.size() % 192 || expected_states.size() != input.size()/192*512 || expected_values.size()!=expected_states.size()) return 2;
    size_t tiles = input.size()/192;
    uint32_t* packed; uint16_t *states,*values;
    HIP(hipMalloc(&packed,input.size())); HIP(hipMalloc(&states,expected_states.size())); HIP(hipMalloc(&values,expected_values.size()));
    HIP(hipMemcpy(packed,input.data(),input.size(),hipMemcpyHostToDevice));
    validate_states<6><<<(tiles+7)/8,256>>>(packed,states,values,tiles,2);
    HIP(hipGetLastError());
    std::vector<char> got_states(expected_states.size()),got_values(expected_values.size());
    HIP(hipMemcpy(got_states.data(),states,got_states.size(),hipMemcpyDeviceToHost));
    HIP(hipMemcpy(got_values.data(),values,got_values.size(),hipMemcpyDeviceToHost));
    bool match = got_states == expected_states && got_values == expected_values;
    std::printf("{\"fixture\":\"EAQ_v1_wkv\",\"tiles\":%zu,\"weights\":%zu,\"bit_exact\":%s}\n",tiles,tiles*256,match?"true":"false");
    HIP(hipFree(packed)); HIP(hipFree(states)); HIP(hipFree(values)); return match?0:2;
}
int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1],"--fixture") == 0) return check_fixture(argv[2],argv[3],argv[4]);
    size_t bytes = argc > 1 ? std::strtoull(argv[1],nullptr,10) * 1024 * 1024 : 256 * 1024 * 1024;
    int repeats = argc > 2 ? std::atoi(argv[2]) : 11;
    if (bytes < 1024 * 1024 || bytes > size_t(4)*1024*1024*1024 || repeats < 1 || repeats > 100) return 2;
    hipDeviceProp_t properties; HIP(hipGetDeviceProperties(&properties,0));
    std::printf("{\"device\":\"%s\",\"arch\":\"%s\",\"cu\":%d,\"wave\":%d}\n",properties.name,properties.gcnArchName,properties.multiProcessorCount,properties.warpSize);
    for (int cb = 0; cb < 3; ++cb) { run<2>(bytes,cb,repeats); run<3>(bytes,cb,repeats); run<4>(bytes,cb,repeats); run<5>(bytes,cb,repeats); run<6>(bytes,cb,repeats); run<8>(bytes,cb,repeats); }
}
