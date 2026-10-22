// Execute the unchanged API implementation with the existing CPU HIP mocks.
// Numeric addresses below describe synthetic intervals, never native GPU proof.
#define main existing_api_contract_main
#include "test_api_contract.cpp"
#undef main
#include "runtime_v14_extent.hpp"
#include <stdexcept>
static void check(bool ok) {if(!ok)throw std::runtime_error("CPU extent assertion failed");}
int main(int argc,char** argv) {
    try {
        const std::string mutation=argc>1?argv[1]:"";
        strix_context_impl context{};strix_weights_impl weight{};
        weight.owner=&context;weight.allocation=reinterpret_cast<void*>(0x10000000);weight.allocation_bytes=4096;
        auto stream=descriptor<strix_stream>();stream.hip_stream=reinterpret_cast<void*>(1);
        for(int n:{5120,32768})for(int m:{2,4,8,64,128,256}) {
            const size_t ybytes=size_t(m)*n*4,xbytes=size_t(m)*1280*4;
            weight.device.input=1280;weight.device.output=n;
            strix_matmul_desc ds[2];
            for(int i=0;i<2;++i){auto& d=ds[i];strix_matmul_desc_init(&d);d.weights=&weight;
                d.input.data=reinterpret_cast<void*>(0x20000000ull+i*0x10000000ull);
                d.output.data=reinterpret_cast<void*>(0x40000000ull+i*0x10000000ull);
                d.input.rank=d.output.rank=2;d.input.shape[0]=d.output.shape[0]=m;
                d.input.shape[1]=d.input.strides[0]=1280;d.output.shape[1]=d.output.strides[0]=n;
                d.input.strides[1]=d.output.strides[1]=1;d.input.buffer_bytes=xbytes;d.output.buffer_bytes=ybytes;
            }
            auto g=descriptor<strix_grouped_moe_desc>();g.groups=ds;g.group_count=2;
            g.workspace=reinterpret_cast<void*>(0x60000000);g.workspace_bytes=size_t(m)*1024;
            fake::buffers.clear();
            for(auto& d:ds){fake::buffers[reinterpret_cast<uintptr_t>(d.input.data)]=xbytes;fake::buffers[reinterpret_cast<uintptr_t>(d.output.data)]=ybytes;}
            fake::buffers[0x60000000]=g.workspace_bytes;
            const auto original=ds[1];const auto launches=fake::launches;
            uint64_t need=0;
            ds[1].output.reserved[0]=1;
            check(strix_grouped_moe(&context,&g,&stream)==(mutation=="descriptor"?STRIX_ERROR_INSUFFICIENT_BUFFER:STRIX_ERROR_INVALID_ARGUMENT));ds[1]=original;
            ds[1].output.buffer_bytes=4;
            check(strix_grouped_moe(&context,&g,&stream)==(mutation=="declared"?STRIX_ERROR_INVALID_ARGUMENT:STRIX_ERROR_INSUFFICIENT_BUFFER));ds[1]=original;
            // V3: a tiny physical allocation whose hypothetical full output
            // crosses the first group's input. The query fails before storage.
            uintptr_t tiny=0x20000000ull-ybytes/2;
            fake::buffers[tiny]=516;ds[1].output.data=reinterpret_cast<void*>(tiny+256);
            check(strix_grouped_moe_workspace_bytes(&context,&g,&need)==STRIX_ERROR_INVALID_ARGUMENT);
            check(strix_grouped_moe(&context,&g,&stream)==(mutation=="alias"?STRIX_ERROR_INSUFFICIENT_BUFFER:STRIX_ERROR_INVALID_ARGUMENT));
            check(device_buffer(&context,ds[1].output.data,ybytes,4)==STRIX_ERROR_INSUFFICIENT_BUFFER);
            fake::buffers.erase(tiny);
            // V4: bounded padded allocation, tail with 4 payload bytes and 256
            // suffix guard bytes. No alias; query succeeds; storage alone fails.
            const uintptr_t base=0x70000000;const size_t requested=ybytes+512;
            fake::buffers[base]=requested;uintptr_t pointer=0;size_t remaining=0;
            check(v14_extent::tail(base,requested,base,requested,256,ybytes,pointer,remaining));
            check(remaining==260 && !v14_extent::overlap(pointer,ybytes,0x60000000,g.workspace_bytes));
            ds[1].output.data=reinterpret_cast<void*>(pointer);
            check(strix_grouped_moe_workspace_bytes(&context,&g,&need)==STRIX_SUCCESS && need==g.workspace_bytes);
            check(strix_grouped_moe(&context,&g,&stream)==(mutation=="extent"?STRIX_ERROR_INVALID_ARGUMENT:STRIX_ERROR_INSUFFICIENT_BUFFER));
            // Mutations of the fixture must be rejected before capture.
            size_t unused=0;uintptr_t p=0;
            check(!v14_extent::tail(base,requested,base,requested+ybytes,256,ybytes,p,unused));
            check(!v14_extent::tail(base,requested,base+4,requested,256,ybytes,p,unused));
            check(v14_extent::overlap(pointer,ybytes,pointer+4,4));
            check(fake::launches==launches);
            std::cout<<"PASS CPU_mock M="<<m<<" N="<<n<<" v3_alias=INVALID_ARGUMENT direct_extent=INSUFFICIENT_BUFFER v4_query=SUCCESS v4_extent=INSUFFICIENT_BUFFER launches=0\n";
        }
        fake::buffers.clear();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
