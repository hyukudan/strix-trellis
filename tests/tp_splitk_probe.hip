// Reuse the established fixture, graph, canary and timing harness.
#define main rank_v03_probe_main
#include "rank_fused_probe.hip"
#undef main
#include "../src/tp_splitk.hpp"
#include "../src/hadamard.hpp"
#include "../src/tp_activation.hpp"
#include "../src/gemv_lds.hpp"
#include <filesystem>
namespace {
using namespace strix_trellis;
// Four resident CTAs perturb wave placement without occupying the whole GPU.
// The stop store is issued on the test stream, which retains free WGPs. This
// load is only used in correctness tests, never in a timed campaign.
__global__ void stress_background(uint32_t* stop, volatile float* output) {
    const int i=int(blockIdx.x)*256+threadIdx.x;
    if(!threadIdx.x)__hip_atomic_store(stop+1+blockIdx.x,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
    float v=float(i+1)*0.0001f;
    while(!__hip_atomic_load(stop,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)) {
        #pragma unroll
        for(int j=0;j<32;++j)v=fmaf(v,0.99991f,0.000013f);
        output[i]=v;
    }
}
__global__ void stress_background_stop(uint32_t* stop) {
    __hip_atomic_store(stop,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
struct BackgroundLoad {
    Runtime& rt;
    Buffer stop,output;
    hipStream_t stream=nullptr;
    bool running=false;
    explicit BackgroundLoad(Runtime& r):rt(r),stop(r,256),output(r,4*256*4) {
        hip_ok(hipMemsetAsync(stop.data,0,stop.bytes,rt.stream));
        hip_ok(hipStreamSynchronize(rt.stream));
        hip_ok(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        stress_background<<<4,256,0,stream>>>(reinterpret_cast<uint32_t*>(stop.data),reinterpret_cast<float*>(output.data));
        const auto error=hipGetLastError();
        if(error!=hipSuccess){(void)hipStreamDestroy(stream);stream=nullptr;hip_ok(error);}
        running=true;
        bool ready=false;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        do {
            const auto flags=stop.download();uint32_t values[5]{};std::memcpy(values,flags.data(),sizeof(values));
            ready=values[1] && values[2] && values[3] && values[4];
        }while(!ready && std::chrono::steady_clock::now()<deadline);
        if(!ready){finish();require(false,"Background CTAs did not become resident");}
    }
    void finish() {
        if(!running)return;
        stress_background_stop<<<1,1,0,rt.stream>>>(reinterpret_cast<uint32_t*>(stop.data));
        hip_ok(hipGetLastError());hip_ok(hipStreamSynchronize(stream));
        running=false;hip_ok(hipStreamDestroy(stream));stream=nullptr;
        stop.canaries();output.canaries();
    }
    ~BackgroundLoad() {
        if(running){stress_background_stop<<<1,1,0,rt.stream>>>(reinterpret_cast<uint32_t*>(stop.data));(void)hipStreamSynchronize(stream);}
        if(stream)(void)hipStreamDestroy(stream);
    }
};
struct SplitPlan {
    Runtime& rt;
    Buffer workspace;
    strix_tp_dense_plan handle = nullptr;
    SplitPlan(Runtime& r, const strix_matmul_desc& source, unsigned tile, unsigned mode)
        : rt(r), workspace(r, needed(r,source,tile)) {
        auto d = source; d.workspace = workspace.bytes ? workspace.data : nullptr; d.workspace_bytes = workspace.bytes;
        api_ok(strix_tp_dense_plan_create_splitk(rt.context,&d,&rt.stream_desc,tile,mode,&handle));
    }
    static size_t needed(Runtime& r, const strix_matmul_desc& d, unsigned tile) {
        uint64_t bytes = 0; api_ok(strix_tp_dense_workspace_bytes(r.context,&d,tile,&bytes)); return size_t(bytes);
    }
    ~SplitPlan() { (void)hipStreamSynchronize(rt.stream); if (handle) (void)strix_tp_dense_plan_destroy(handle); }
    void call() { api_ok(strix_tp_dense_plan_submit(handle)); }
};
struct RawWeight {
    Buffer packed, suh, svh, bias;
    DeviceWeights device{};
    RawWeight(Runtime& r, const Fixture& f) : packed(r,f.sections.at("trellis").size()),
        suh(r,f.sections.at("suh").size()),svh(r,f.sections.at("svh").size()),bias(r,f.sections.at("bias").size()) {
        const char* names[] = {"trellis","suh","svh","bias"}; Buffer* b[] = {&packed,&suh,&svh,&bias};
        for (int i=0;i<4;++i) b[i]->upload(f.sections.at(names[i]).data(),b[i]->bytes);
        device = {reinterpret_cast<const uint32_t*>(packed.data),reinterpret_cast<const uint16_t*>(suh.data),
            reinterpret_cast<const uint16_t*>(svh.data),reinterpret_cast<const float*>(bias.data),int(f.input),int(f.output),int(f.bits),2};
    }
};
struct DotCase { uint16_t w0,x0,w1,x1; float acc, expected; };
__global__ void fdot_cases(const DotCase* in, float* out, int count) {
    const int i=int(blockIdx.x)*256+threadIdx.x;
    if (i>=count) return;
    using Pair = _Float16 __attribute__((ext_vector_type(2)));
    const auto c = in[i];
    Pair w{__builtin_bit_cast(_Float16,c.w0),__builtin_bit_cast(_Float16,c.w1)};
    Pair x{__builtin_bit_cast(_Float16,c.x0),__builtin_bit_cast(_Float16,c.x1)};
    out[i] = __builtin_amdgcn_fdot2(w,x,c.acc,false);
}
void dot_gate(Runtime& rt,const std::string& dir) {
    std::ifstream file(dir+"/fdot2.bin",std::ios::binary);
    require(bool(file),"Missing exact-dyadic fdot2 fixture");
    std::vector<DotCase> cases(4096); file.read(reinterpret_cast<char*>(cases.data()),cases.size()*sizeof(DotCase));
    require(bool(file),"Short fdot2 fixture");
    Buffer input(rt,cases.size()*sizeof(DotCase)),output(rt,cases.size()*4);
    input.upload(cases.data(),input.bytes);
    fdot_cases<<<16,256,0,rt.stream>>>(reinterpret_cast<const DotCase*>(input.data),reinterpret_cast<float*>(output.data),int(cases.size()));
    hip_ok(hipGetLastError()); const auto result = output.download();
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto results=std::filesystem::path(dir).parent_path()/"results";
    std::filesystem::create_directories(results);
    const std::string dump=(results/("fdot-native-"+std::to_string(stamp)+".bin")).string();
    {std::ofstream file(dump,std::ios::binary);file.write(reinterpret_cast<const char*>(result.data()),result.size());require(bool(file),"Cannot preserve native dot outputs");}
    std::printf("TP_FDOT_DUMP path=%s\n",dump.c_str());
    size_t mismatches=0;
    for (size_t i=0;i<cases.size();++i) if(std::memcmp(result.data()+4*i,&cases[i].expected,4)) {
        float value=0;std::memcpy(&value,result.data()+4*i,4);
        if(mismatches++<16)std::printf("TP_FDOT_MISMATCH case=%zu expected=%a actual=%a\n",i,double(cases[i].expected),double(value));
    }
    std::printf("TP_FDOT_SUMMARY cases=%zu ieee_differences=%zu cpu_bit_emulator_qualified=0\n",cases.size(),mismatches);
    std::vector<float> actual(cases.size()),reference(cases.size());
    std::memcpy(actual.data(),result.data(),result.size());
    for(size_t i=0;i<cases.size();++i)reference[i]=cases[i].expected;
    compare("dot2_native_precision_characterization",actual,reference);
    std::puts("TP_FDOT_PRECISION_PASS cases=4096 cpu_bit_emulator_qualified=0 exactness_gate=GPU_legacy_P");
}
void core_gate(Runtime& rt,const std::string& dir) {
    std::ifstream file(dir+"/core-oracle.bin",std::ios::binary);require(bool(file),"Missing split core oracle");
    uint32_t header[4]{};file.read(reinterpret_cast<char*>(header),sizeof(header));
    require(header[0]==512 && header[1]==16 && header[2]==2 && header[3]==2,"Invalid core oracle header");
    for(unsigned b=0;b<2;++b) {
        uint32_t bits=0;file.read(reinterpret_cast<char*>(&bits),4);require(bits==4+b,"Invalid core oracle rate");
        std::vector<unsigned char> packed(size_t(512)*16*bits/8),act(2*512*2);
        file.read(reinterpret_cast<char*>(packed.data()),packed.size());file.read(reinterpret_cast<char*>(act.data()),act.size());
        Buffer weights(rt,packed.size()),a(rt,act.size());weights.upload(packed.data(),packed.size());a.upload(act.data(),act.size());
        DeviceWeights w{};w.trellis=reinterpret_cast<const uint32_t*>(weights.data);w.input=512;w.output=16;w.bits=int(bits);w.codebook=2;
        for(int s:{4,16,32}) {
            std::vector<unsigned char> expected(size_t(s)*2*16*4);file.read(reinterpret_cast<char*>(expected.data()),expected.size());require(bool(file),"Short core oracle");
            Buffer p(rt,expected.size());hip_ok(launch_gemv_lds_variant(w,reinterpret_cast<const _Float16*>(a.data),reinterpret_cast<float*>(p.data),2,2,s,rt.stream,0));
            const auto actual_bytes=p.download();std::vector<float> actual(expected.size()/4),reference(expected.size()/4);
            std::memcpy(actual.data(),actual_bytes.data(),expected.size());std::memcpy(reference.data(),expected.data(),expected.size());
            compare("core_CPU_precision_reference",actual,reference);
            p.canaries();std::printf("TP_CORE_CPU_PRECISION_PASS bits=%u S=%d rows=2 bit_identity_gate=separate_GPU_legacy_P\n",bits,s);
        }
    }
}
__global__ void dot_calibration(const DotCase* cases,DotCase* echo,float* out) {
    const int i=int(blockIdx.x)*256+threadIdx.x;
    const auto c=cases[i];echo[i]=c;
    using Pair=_Float16 __attribute__((ext_vector_type(2)));
    const Pair w{__builtin_bit_cast(_Float16,c.w0),__builtin_bit_cast(_Float16,c.w1)};
    const Pair x{__builtin_bit_cast(_Float16,c.x0),__builtin_bit_cast(_Float16,c.x1)};
    uint32_t z=0;asm volatile("" : "+v"(z));
    const _Float16 zero=__builtin_bit_cast(_Float16,uint16_t(z));
    out[i*8+0]=__builtin_amdgcn_fdot2(w,x,c.acc,false);
    out[i*8+1]=__builtin_amdgcn_fdot2(w,x,float(z),false);
    out[i*8+2]=__builtin_amdgcn_fdot2(Pair{w[0],zero},Pair{x[0],zero},float(z),false);
    out[i*8+3]=__builtin_amdgcn_fdot2(Pair{zero,w[1]},Pair{zero,x[1]},float(z),false);
    const float low=fmaf(float(w[0]),float(x[0]),c.acc);
    const float high=fmaf(float(w[1]),float(x[1]),c.acc);
    out[i*8+4]=fmaf(float(w[1]),float(x[1]),low);
    out[i*8+5]=fmaf(float(w[0]),float(x[0]),high);
    out[i*8+6]=c.acc;
    out[i*8+7]=low;
}
void calibrate_dot(Runtime& rt,const std::string& dir) {
    std::ifstream file(dir+"/fdot2.bin",std::ios::binary);
    std::vector<DotCase> cases(4096);file.read(reinterpret_cast<char*>(cases.data()),cases.size()*sizeof(DotCase));require(bool(file),"Missing calibration cases");
    Buffer in(rt,cases.size()*sizeof(DotCase)),echo(rt,in.bytes),out(rt,cases.size()*8*4);in.upload(cases.data(),in.bytes);
    dot_calibration<<<16,256,0,rt.stream>>>(reinterpret_cast<const DotCase*>(in.data),reinterpret_cast<DotCase*>(echo.data),reinterpret_cast<float*>(out.data));hip_ok(hipGetLastError());
    const auto echoed=echo.download();require(!std::memcmp(echoed.data(),cases.data(),echoed.size()),"Calibration input echo changed bytes");
    const auto result=out.download();const auto results=std::filesystem::path(dir).parent_path()/"results";std::filesystem::create_directories(results);
    const auto path=results/("dot-calibration-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    std::ofstream dump(path,std::ios::binary);dump.write(reinterpret_cast<const char*>(result.data()),result.size());require(bool(dump),"Cannot preserve calibration output");
    std::printf("TP_DOT_CALIBRATION path=%s cases=4096 columns=dot,zero_acc,low_product,high_product,low_high_fma,high_low_fma,acc,low_fma input_echo=bytes\n",path.string().c_str());
}
strix_matmul_desc descriptor(Weight& w, View& x, View& y, float alpha=1,float beta=0) {
    strix_matmul_desc d; strix_matmul_desc_init(&d); d.weights=w.handle; d.input=x.descriptor();d.output=y.descriptor();d.alpha=alpha;d.beta=beta;return d;
}
void dense_edge_case(Runtime& rt,const Fixture& f,const std::string& site) {
    const std::vector<float> no_bias;
    Weight w(rt,f,&no_bias);
    const auto x=f.values<float>("x"),old=f.values<float>("old");
    struct Epilogue { float alpha,beta; };
    const Epilogue cases[]={{1,-0.0f},{0,0},{-0.0f,0},{-.5f,.25f}};
    for(unsigned rows:{1u,8u}) {
        View input(rt,rows,f.input),output(rt,rows,f.output),bx(rt,rows,f.input,false),reference(rt,rows,f.output,false);
        input.reset(x);bx.reset(x);
        for(const auto& c:cases) {
            auto d=descriptor(w,input,output,c.alpha,c.beta),b=descriptor(w,bx,reference,c.alpha,c.beta);
            uint64_t bytes=0;api_ok(strix_matmul_workspace_bytes(rt.context,&b,&bytes));
            Buffer baseline_workspace(rt,size_t(bytes));b.workspace=baseline_workspace.data;b.workspace_bytes=bytes;
            LegacyPlan baseline(rt,b);
            reference.reset(old,c.beta==0);api_ok(strix_linear_plan_submit(baseline.handle));const auto expected=reference.values();
            for(unsigned mode:{0u,1u}) {
                SplitPlan plan(rt,d,rows,mode);output.reset(old,c.beta==0);plan.call();
                exact(output.values(),expected,"Absent-bias/zero epilogue differs from legacy bytes");
                {Graph graph;graph.capture(rt,[&]{plan.call();},mode && plan.workspace.bytes?2:1);
                    output.reset(old,c.beta==0);graph.replay(rt);
                    exact(output.values(),expected,"Absent-bias/zero graph differs from legacy bytes");
                }
                if(plan.workspace.bytes) {
                    TpSplitLayout layout{};require(tp_split_layout(int(f.input),int(f.output),int(rows),int(rows),layout),"Missing edge layout");
                    const auto memory=plan.workspace.download();
                    for(size_t i=layout.counter_offset;i<layout.bytes;++i)require(memory[i]==0,"Edge counter not reset");
                }
                plan.workspace.canaries();input.unchanged();bx.unchanged();
                std::printf("TP_EDGE_PASS site=%s bits=%u rows=%u mode=%u no_bias=1 alpha=%a beta=%a nan_old=%d Y=bytes graph_nodes=%d\n",
                    site.c_str(),f.bits,rows,mode,double(c.alpha),double(c.beta),int(c.beta==0),mode && plan.workspace.bytes?2:1);
            }
        }
    }
}
void split_case(Runtime& rt,const Fixture& f,const std::string& site,bool identity_only) {
    Weight w(rt,f); RawWeight raw(rt,f);
    const auto x=f.values<float>("x"),old=f.values<float>("old"),ref=f.values<float>("ref");
    const auto bounds=f.values<double>("bound");
    for (unsigned rows=1;rows<=(identity_only?1u:8u);++rows) {
        View input(rt,rows,f.input),output(rt,rows,f.output), bx(rt,rows,f.input,false),by(rt,rows,f.output,false);
        input.reset(x);bx.reset(x);by.reset(old,true);
        auto bd=descriptor(w,bx,by);
        uint64_t wb=0;api_ok(strix_matmul_workspace_bytes(rt.context,&bd,&wb));
        Buffer bws(rt,wb);bd.workspace=bws.data;bd.workspace_bytes=wb;LegacyPlan baseline(rt,bd);
        api_ok(strix_linear_plan_submit(baseline.handle)); const auto expected=by.values();
        auto d=descriptor(w,input,output);
        { DensePlan direct(rt,d,rows); output.reset(old,true);api_ok(strix_tp_dense_plan_submit(direct.handle));
          const auto original=output.values();const bool same=!std::memcmp(expected.data(),original.data(),expected.size()*4);
          std::printf("TP_V03_IDENTITY site=%s bits=%u rows=%u identical=%d legacy_s=%d\n",site.c_str(),f.bits,rows,int(same),tp_split_count(int(f.input),int(f.output)));
          require(same == !tp_split_largek(int(f.input),int(f.output)),"Unexpected 0.3/legacy identity experiment"); }
        if(identity_only) continue;
        bd.alpha=-.5f;bd.beta=.25f;LegacyPlan epilogue_baseline(rt,bd);by.reset(old);api_ok(strix_linear_plan_submit(epilogue_baseline.handle));const auto expected_epilogue=by.values();
        const auto legacy_workspace=bws.download();
        for(unsigned tile=1;tile<=rows;++tile) for(unsigned mode : {0u,1u}) {
            SplitPlan p(rt,d,tile,mode);output.reset(old,true);p.call();const auto actual=output.values();
            exact(actual,expected,"Split-K output differs from legacy bytes");
            const std::string label="site="+site+" bits="+std::to_string(f.bits)+" rows="+std::to_string(rows)+" tile="+std::to_string(tile)+" mode="+std::to_string(mode);
            compare(label,actual,ref,1,0,nullptr,5e-4,3e-5,&bounds);
            TpSplitLayout l{};
            if(tp_split_layout(int(f.input),int(f.output),int(rows),int(tile),l)) {
                const int cache_columns=tp_split_cache_h128(int(f.input))*128;
                Buffer activation(rt,size_t(tp_split_count(int(f.input),int(f.output)))*rows*cache_columns*2);
                TpSplitLaunch launch{};
                launch.member[0]={raw.device,static_cast<const float*>(d.input.data),static_cast<float*>(d.output.data),d.input.strides[0],d.output.strides[0],int(rows),1,0,int(tile)};
                launch.partial[0]=reinterpret_cast<float*>(p.workspace.data);launch.counters=reinterpret_cast<uint32_t*>(p.workspace.data+l.counter_offset);
                launch.activation=reinterpret_cast<uint16_t*>(activation.data);launch.two_kernel=mode!=0;
                hip_ok(launch_tp_split_trusted(launch,rt.stream));exact(output.values(),expected,"Diagnostic output changed bytes");
                const auto a=activation.download(), pw=p.workspace.download();
                const size_t a_bytes=(size_t(rows)*f.input*2+255)&~size_t(255);
                require(!std::memcmp(pw.data(),legacy_workspace.data()+a_bytes,l.partial_bytes),"P differs from legacy partial planes");
                for(int s=0;s<tp_split_count(int(f.input),int(f.output));++s) {
                    const auto part=tp_split_partition(int(f.input),s,tp_split_count(int(f.input),int(f.output)));
                    for(unsigned r=0;r<rows;++r)for(int h=part.first_h128;h<part.end_h128;++h)
                        require(!std::memcmp(a.data()+((size_t(s)*rows+r)*cache_columns+(h-part.first_h128)*128)*2,
                            legacy_workspace.data()+(size_t(r)*f.input+h*128)*2,256),"A differs from input_transform bytes");
                }
                for(size_t i=l.counter_offset;i<l.bytes;++i)require(pw[i]==0,"Counter not reset to zero");
                if(rows==8 && tile==rows)for(int arrival=1;arrival<=tp_split_count(int(f.input),int(f.output));++arrival) {
                    launch.schedule_rotation=unsigned(arrival*(f.output/128));
                    hip_ok(hipMemsetAsync(p.workspace.data,0xff,l.partial_bytes,rt.stream));
                    hip_ok(launch_tp_split_trusted(launch,rt.stream));
                    exact(output.values(),expected,"CTA split arrival permutation changed bytes");
                }
                p.workspace.canaries();activation.canaries();
            }
            {
                Graph graph;graph.capture(rt,[&]{p.call();},mode && p.workspace.bytes?2:1);
                for(int background=0;background<(tile==rows?2:1);++background) {
                    std::unique_ptr<BackgroundLoad> load;
                    if(background)load=std::make_unique<BackgroundLoad>(rt);
                    for(int repeat=0;repeat<(tile==rows?1000:2);++repeat) {
                        if(p.workspace.bytes)hip_ok(hipMemsetAsync(p.workspace.data,0xff,l.partial_bytes,rt.stream));
                        output.reset(old,true);graph.replay(rt);exact(output.values(),expected,"Split-K graph stress changed bytes");
                    }
                    if(load)load->finish();
                    if(p.workspace.bytes) {
                        const auto memory=p.workspace.download();
                        for(size_t i=l.counter_offset;i<l.bytes;++i)
                            require(memory[i]==0,"Counter not reset after graph stress series");
                    }
                }
                auto changed=x;for(auto& value:changed)value=-value;
                bx.reset(changed);by.reset(old,true);api_ok(strix_linear_plan_submit(baseline.handle));const auto changed_ref=by.values();
                input.reset(changed);output.reset(old,true);graph.replay(rt);
                exact(output.values(),changed_ref,"Graph replay did not observe changed X contents");
                input.reset(x);bx.reset(x);
            }
            d.alpha=-.5f;d.beta=.25f;SplitPlan epilogue(rt,d,tile,mode);output.reset(old);epilogue.call();
            exact(output.values(),expected_epilogue,"Split-K alpha/beta differs from legacy bytes");d.alpha=1;d.beta=0;
            input.unchanged();bx.unchanged();p.workspace.canaries();
            if(rows==8 && tile==rows && mode==0) {
                View concurrent(rt,rows,f.output);concurrent.reset(old,true);
                auto other=d;other.output=concurrent.descriptor();
                Buffer other_ws(rt,p.workspace.bytes);other.workspace=other_ws.bytes?other_ws.data:nullptr;other.workspace_bytes=other_ws.bytes;
                hipStream_t second=nullptr;hip_ok(hipStreamCreateWithFlags(&second,hipStreamNonBlocking));
                auto stream=rt.stream_desc;stream.hip_stream=reinterpret_cast<void*>(second);
                hip_ok(hipStreamSynchronize(rt.stream));
                strix_tp_dense_plan second_plan=nullptr;api_ok(strix_tp_dense_plan_create_splitk(rt.context,&other,&stream,tile,mode,&second_plan));
                output.reset(old,true);hip_ok(hipStreamSynchronize(rt.stream));p.call();api_ok(strix_tp_dense_plan_submit(second_plan));
                hip_ok(hipStreamSynchronize(second));exact(concurrent.values(),expected,"Disjoint stream changed output bytes");exact(output.values(),expected,"Shared-read concurrent stream changed output bytes");
                api_ok(strix_tp_dense_plan_destroy(second_plan));hip_ok(hipStreamDestroy(second));other_ws.canaries();
            }
            std::printf("TP_SPLIT_PASS %s A=%s P=%s Y=bytes counters=zero graph_nodes=%d stress_replays=%d background_replays=%d\n",label.c_str(),p.workspace.bytes?"bytes":"retained_v03",p.workspace.bytes?"bytes":"retained_v03",mode&&p.workspace.bytes?2:1,tile==rows?1000:2,tile==rows?1000:0);
        }
    }
}
void pair_case(Runtime& rt,const Fixture& f0,const Fixture& f1,bool swiglu,bool equal) {
    require(f0.sections.at("x")==f1.sections.at("x"),"Pair inputs differ");
    require((f0.sections.at("suh")==f1.sections.at("suh"))==equal,"Pair scale identity fixture is wrong");
    Weight w0(rt,f0),w1(rt,f1);RawWeight raw0(rt,f0),raw1(rt,f1);
    const auto x=f0.values<float>("x"),old0=f0.values<float>("old"),old1=f1.values<float>("old");
    for(unsigned rows=1;rows<=8;++rows) {
        View input(rt,rows,5120),y0(rt,rows,f0.output),y1(rt,rows,f1.output),hidden(rt,rows,1152);
        input.reset(x);y0.reset(old0,true);y1.reset(old1,true);
        auto d0=descriptor(w0,input,y0),d1=descriptor(w1,input,y1);
        SplitPlan individual0(rt,d0,rows,0),individual1(rt,d1,rows,0);
        individual0.call();individual1.call();const auto expected0=y0.values(),expected1=y1.values();
        std::vector<float> hidden_ref(expected0.size());
        if(swiglu)for(size_t i=0;i<hidden_ref.size();++i) {
            const float g=expected0[i],e=std::exp(g<0?g:-g),sigmoid=g<0?e/(1+e):1/(1+e);
            hidden_ref[i]=(g*sigmoid)*expected1[i];
        }
        for(unsigned tile=1;tile<=rows;++tile)for(unsigned mode:{0u,1u}) {
            strix_tp_dense_pair_desc d;strix_tp_dense_pair_desc_init(&d);
            d.input=input.descriptor();d.member[0]={w0.handle,y0.descriptor(),1,0};d.member[1]={w1.handle,y1.descriptor(),1,0};
            d.rows_per_block=tile;d.strategy=mode;
            strix_tp_swiglu_desc s;strix_tp_swiglu_desc_init(&s);
            s.input=d.input;s.gate=w0.handle;s.up=w1.handle;s.hidden_output=hidden.descriptor();s.rows_per_block=tile;s.strategy=mode;
            uint64_t bytes=0;
            if(swiglu)api_ok(strix_tp_swiglu_workspace_bytes(rt.context,&s,&bytes));
            else api_ok(strix_tp_dense_pair_workspace_bytes(rt.context,&d,&bytes));
            Buffer ws(rt,size_t(bytes)),diagnostic0(rt,size_t(rows)*1152*4),diagnostic1(rt,size_t(rows)*1152*4);
            d.workspace=s.workspace=ws.data;d.workspace_bytes=s.workspace_bytes=bytes;
            strix_tp_dense_pair_plan plan=nullptr;
            if(swiglu) {
                s.diagnostic_gate=reinterpret_cast<float*>(diagnostic0.data);s.diagnostic_up=reinterpret_cast<float*>(diagnostic1.data);s.diagnostic_bytes=diagnostic0.bytes;
                api_ok(strix_tp_swiglu_plan_create(rt.context,&s,&rt.stream_desc,&plan));
            } else api_ok(strix_tp_dense_pair_plan_create(rt.context,&d,&rt.stream_desc,&plan));
            auto call=[&]{api_ok(strix_tp_dense_pair_plan_submit(plan));};
            call();
            const auto first_hidden=swiglu?hidden.values():std::vector<float>{};
            auto check=[&]{
                if(swiglu) {
                    const auto g=diagnostic0.download(),u=diagnostic1.download();
                    require(!std::memcmp(g.data(),expected0.data(),g.size()),"Fused gate differs from individual bytes");
                    require(!std::memcmp(u.data(),expected1.data(),u.size()),"Fused up differs from individual bytes");
                    const auto h=hidden.values();exact(h,first_hidden,"SwiGLU hidden changed bytes across repeated/rotated submits");
                    compare("split_swiglu_hidden",h,hidden_ref);
                } else {exact(y0.values(),expected0,"q_a pair bytes differ");exact(y1.values(),expected1,"kv pair bytes differ");}
            };
            check();
            {Graph graph;graph.capture(rt,call,mode?2:1);
                for(int background=0;background<(tile==rows?2:1);++background) {
                    std::unique_ptr<BackgroundLoad> load;
                    if(background)load=std::make_unique<BackgroundLoad>(rt);
                    for(int repeat=0;repeat<(tile==rows?1000:2);++repeat){
                        graph.replay(rt);
                        if(swiglu){
                            const auto g=diagnostic0.download(),u=diagnostic1.download();
                            require(!std::memcmp(g.data(),expected0.data(),g.size()),"SwiGLU graph gate changed bytes");
                            require(!std::memcmp(u.data(),expected1.data(),u.size()),"SwiGLU graph up changed bytes");
                            exact(hidden.values(),first_hidden,"SwiGLU graph hidden changed bytes");
                        }else{exact(y0.values(),expected0,"Pair graph q_a changed bytes");exact(y1.values(),expected1,"Pair graph kv changed bytes");}
                    }
                    if(load)load->finish();
                    const auto memory=ws.download();
                    for(size_t i=size_t(rows)*147456;i<bytes;++i)
                        require(memory[i]==0,"Pair counter not reset after graph stress series");
                }
            }
            if(rows==8 && tile==rows) {
                TpSplitLaunch launch{};launch.members=2;launch.swiglu=swiglu;launch.two_kernel=mode!=0;
                const auto out0=swiglu?hidden.descriptor():y0.descriptor(),out1=swiglu?hidden.descriptor():y1.descriptor();
                launch.member[0]={raw0.device,static_cast<const float*>(d.input.data),static_cast<float*>(out0.data),d.input.strides[0],out0.strides[0],int(rows),1,0,int(tile)};
                launch.member[1]={raw1.device,static_cast<const float*>(d.input.data),static_cast<float*>(out1.data),d.input.strides[0],out1.strides[0],int(rows),1,0,int(tile)};
                launch.partial[0]=reinterpret_cast<float*>(ws.data);
                launch.partial[1]=reinterpret_cast<float*>(ws.data+size_t(tp_split_count(5120,int(f0.output)))*rows*f0.output*4);
                launch.counters=reinterpret_cast<uint32_t*>(ws.data+size_t(rows)*147456);
                if(swiglu){launch.diagnostic[0]=reinterpret_cast<float*>(diagnostic0.data);launch.diagnostic[1]=reinterpret_cast<float*>(diagnostic1.data);}
                for(unsigned arrival=1;arrival<=32;++arrival) {
                    launch.schedule_rotation=arrival*(swiglu?9u:10u);
                    hip_ok(hipMemsetAsync(ws.data,0xff,size_t(rows)*147456,rt.stream));
                    hip_ok(launch_tp_split_trusted(launch,rt.stream));
                    hip_ok(hipStreamSynchronize(rt.stream));
                    check();
                }
                check();
            }
            if(!swiglu) {
                api_ok(strix_tp_dense_pair_plan_destroy(plan));plan=nullptr;
                d.member[0].alpha=d.member[1].alpha=-.5f;d.member[0].beta=d.member[1].beta=.25f;
                auto e0=d0,e1=d1;e0.alpha=e1.alpha=-.5f;e0.beta=e1.beta=.25f;
                SplitPlan ep0(rt,e0,tile,mode),ep1(rt,e1,tile,mode);
                y0.reset(old0);y1.reset(old1);ep0.call();ep1.call();const auto r0=y0.values(),r1=y1.values();
                api_ok(strix_tp_dense_pair_plan_create(rt.context,&d,&rt.stream_desc,&plan));y0.reset(old0);y1.reset(old1);call();
                exact(y0.values(),r0,"Paired q_a alpha/beta bytes differ");exact(y1.values(),r1,"Paired kv alpha/beta bytes differ");
            }
            const auto memory=ws.download();const size_t partial=size_t(rows)*147456;
            for(size_t i=partial;i<bytes;++i)require(memory[i]==0,"Pair counters not reset");
            api_ok(strix_tp_dense_pair_plan_destroy(plan));ws.canaries();input.unchanged();
            std::printf("TP_PAIR_PASS kind=%s bits0=%u bits1=%u suh_equal=%d rows=%u tile=%u mode=%u graph_nodes=%u workspace=%llu stress_replays=%d background_replays=%d\n",
                swiglu?"swiglu":"q_a_kv",f0.bits,f1.bits,int(equal),rows,tile,mode,mode?2:1,static_cast<unsigned long long>(bytes),tile==rows?1000:2,tile==rows?1000:0);
        }
    }
}
void paired_measure(Runtime& rt,const std::string& label,const std::function<void()>& baseline,
    const std::function<void()>& candidate,const std::function<void()>& v03,int repetitions,const char* protocol) {
    for(int pair=0;pair<3;++pair) {
        const auto prefix=label+" pair="+std::to_string(pair)+" order="+(pair%2?"candidate_first":"baseline_first");
        if(pair%2) {
            measure(rt,prefix+" producer=splitk",candidate,repetitions,protocol);
            measure(rt,prefix+" producer=baseline",baseline,repetitions,protocol);
        } else {
            measure(rt,prefix+" producer=baseline",baseline,repetitions,protocol);
            measure(rt,prefix+" producer=splitk",candidate,repetitions,protocol);
        }
        measure(rt,prefix+" producer=v03",v03,repetitions,protocol);
    }
}
void bench_dense(Runtime& rt,const Fixture& f,const std::string& site,int repetitions,bool cold) {
    const auto x=f.values<float>("x"),old=f.values<float>("old");
    for(unsigned rows=1;rows<=8;++rows) {
        if(cold && rows!=1 && rows!=8)continue;
        View input(rt,rows,f.input,false),output(rt,rows,f.output,false);input.reset(x);output.reset(old,true);
        const size_t packed=size_t(f.input)*f.output*f.bits/8;
        const size_t count=cold?(256*1024*1024+packed-1)/packed:1;
        std::vector<std::unique_ptr<Weight>> weights;
        std::vector<std::unique_ptr<LegacyPlan>> baselines;
        std::vector<std::unique_ptr<DensePlan>> originals;
        std::vector<strix_tp_dense_plan> candidates;
        weights.emplace_back(std::make_unique<Weight>(rt,f));
        auto first=descriptor(*weights[0],input,output);
        uint64_t basebytes=0,newbytes=0;api_ok(strix_matmul_workspace_bytes(rt.context,&first,&basebytes));
        api_ok(strix_tp_dense_workspace_bytes(rt.context,&first,rows,&newbytes));
        Buffer bws(rt,size_t(basebytes)),ws(rt,size_t(newbytes));
        for(size_t i=0;i<count;++i) {
            if(i)weights.emplace_back(std::make_unique<Weight>(rt,f));
            auto d=descriptor(*weights[i],input,output);
            originals.emplace_back(std::make_unique<DensePlan>(rt,d,rows));
            d.workspace=bws.data;d.workspace_bytes=bws.bytes;baselines.emplace_back(std::make_unique<LegacyPlan>(rt,d));
            d.workspace=newbytes?ws.data:nullptr;d.workspace_bytes=newbytes;
            strix_tp_dense_plan plan=nullptr;api_ok(strix_tp_dense_plan_create_splitk(rt.context,&d,&rt.stream_desc,rows,0,&plan));candidates.push_back(plan);
        }
        size_t bi=0,ci=0,oi=0;
        auto baseline=[&]{api_ok(strix_linear_plan_submit(baselines[bi++%count]->handle));};
        auto candidate=[&]{api_ok(strix_tp_dense_plan_submit(candidates[ci++%count]));};
        auto v03=[&]{api_ok(strix_tp_dense_plan_submit(originals[oi++%count]->handle));};
        baseline();const auto expected=output.values();candidate();exact(output.values(),expected,"Benchmark split candidate failed byte gate");
        const std::string label="kind=dense site="+site+" bits="+std::to_string(f.bits)+" rows="+std::to_string(rows)+" tile="+std::to_string(rows)+" packed_bytes="+std::to_string(packed)+" pool="+std::to_string(count);
        std::printf("TP_PLAN %s strategy=%s S=%d grid=%d row_blocks=1 threads=%d launches=1 workspace=%llu\n",label.c_str(),newbytes?"splitk":"v03",tp_split_count(int(f.input),int(f.output)),int(f.output/128)*(newbytes?tp_split_count(int(f.input),int(f.output)):1),newbytes?256:512,static_cast<unsigned long long>(newbytes));
        // Each cold sweep traverses the entire resident disjoint packed pool.
        const int reps=cold?int(count):repetitions;
        paired_measure(rt,label,baseline,candidate,v03,reps,cold?"resident_disjoint_pool":"hot_fixed_addresses");
        candidate();exact(output.values(),expected,"Timed split output changed");
        for(auto p:candidates)api_ok(strix_tp_dense_plan_destroy(p));
        ws.canaries();bws.canaries();input.unchanged();
        // Baselines/originals retain weights until their destruction here.
        baselines.clear();originals.clear();weights.clear();
    }
}
__global__ void reference_swiglu(const float* g,const float* u,float* out,uint64_t gs,uint64_t us,uint64_t os,int rows) {
    const int col=int(blockIdx.x)*256+threadIdx.x,row=blockIdx.y;
    if(col<1152 && row<rows)out[size_t(row)*os+col]=tp_stable_swiglu(g[size_t(row)*gs+col],u[size_t(row)*us+col]);
}
void bench_pair(Runtime& rt,const Fixture& f0,const Fixture& f1,bool swiglu,int repetitions,bool cold) {
    Weight w0(rt,f0),w1(rt,f1);
    const auto x=f0.values<float>("x"),old0=f0.values<float>("old"),old1=f1.values<float>("old");
    for(unsigned rows=1;rows<=8;++rows) {
        if(cold && rows!=1 && rows!=8)continue;
        View input(rt,rows,5120,false),y0(rt,rows,f0.output,false),y1(rt,rows,f1.output,false),hidden(rt,rows,1152,false);
        input.reset(x);y0.reset(old0,true);y1.reset(old1,true);
        auto d0=descriptor(w0,input,y0),d1=descriptor(w1,input,y1);
        DensePlan old0plan(rt,d0,rows),old1plan(rt,d1,rows);
        uint64_t wb0=0,wb1=0;api_ok(strix_matmul_workspace_bytes(rt.context,&d0,&wb0));api_ok(strix_matmul_workspace_bytes(rt.context,&d1,&wb1));
        Buffer bws(rt,size_t(std::max(wb0,wb1)));d0.workspace=d1.workspace=bws.data;d0.workspace_bytes=d1.workspace_bytes=bws.bytes;
        LegacyPlan base0(rt,d0),base1(rt,d1);
        strix_tp_dense_pair_desc d;strix_tp_dense_pair_desc_init(&d);d.input=input.descriptor();
        d.member[0]={w0.handle,y0.descriptor(),1,0};d.member[1]={w1.handle,y1.descriptor(),1,0};
        strix_tp_swiglu_desc s;strix_tp_swiglu_desc_init(&s);s.input=d.input;s.gate=w0.handle;s.up=w1.handle;s.hidden_output=hidden.descriptor();
        uint64_t bytes=0;if(swiglu)api_ok(strix_tp_swiglu_workspace_bytes(rt.context,&s,&bytes));else api_ok(strix_tp_dense_pair_workspace_bytes(rt.context,&d,&bytes));
        Buffer ws(rt,size_t(bytes));s.workspace=d.workspace=ws.data;s.workspace_bytes=d.workspace_bytes=bytes;
        strix_tp_dense_pair_plan plan=nullptr;
        if(swiglu)api_ok(strix_tp_swiglu_plan_create(rt.context,&s,&rt.stream_desc,&plan));else api_ok(strix_tp_dense_pair_plan_create(rt.context,&d,&rt.stream_desc,&plan));
        const size_t packed=(size_t(f0.input)*f0.output*f0.bits+size_t(f1.input)*f1.output*f1.bits)/8;
        const size_t count=cold?(256*1024*1024+packed-1)/packed:1;
        std::vector<std::unique_ptr<Weight>> clones0,clones1;
        std::vector<std::unique_ptr<LegacyPlan>> pool_base0,pool_base1;
        std::vector<std::unique_ptr<DensePlan>> pool_old0,pool_old1;
        std::vector<strix_tp_dense_pair_plan> pool_new{plan};
        for(size_t i=1;i<count;++i) {
            clones0.emplace_back(std::make_unique<Weight>(rt,f0));clones1.emplace_back(std::make_unique<Weight>(rt,f1));
            auto a0=descriptor(*clones0.back(),input,y0),a1=descriptor(*clones1.back(),input,y1);
            pool_old0.emplace_back(std::make_unique<DensePlan>(rt,a0,rows));pool_old1.emplace_back(std::make_unique<DensePlan>(rt,a1,rows));
            a0.workspace=a1.workspace=bws.data;a0.workspace_bytes=a1.workspace_bytes=bws.bytes;
            pool_base0.emplace_back(std::make_unique<LegacyPlan>(rt,a0));pool_base1.emplace_back(std::make_unique<LegacyPlan>(rt,a1));
            d.member[0].weights=s.gate=clones0.back()->handle;d.member[1].weights=s.up=clones1.back()->handle;
            strix_tp_dense_pair_plan next=nullptr;
            if(swiglu)api_ok(strix_tp_swiglu_plan_create(rt.context,&s,&rt.stream_desc,&next));else api_ok(strix_tp_dense_pair_plan_create(rt.context,&d,&rt.stream_desc,&next));
            pool_new.push_back(next);
        }
        auto activation=[&]{if(swiglu){reference_swiglu<<<dim3(5,rows),256,0,rt.stream>>>(reinterpret_cast<const float*>(y0.descriptor().data),reinterpret_cast<const float*>(y1.descriptor().data),reinterpret_cast<float*>(hidden.descriptor().data),y0.stride,y1.stride,hidden.stride,int(rows));hip_ok(hipGetLastError());}};
        size_t bi=0,oi=0,ci=0;
        auto baseline=[&]{const size_t i=bi++%count;api_ok(strix_linear_plan_submit(i?pool_base0[i-1]->handle:base0.handle));api_ok(strix_linear_plan_submit(i?pool_base1[i-1]->handle:base1.handle));activation();};
        auto original=[&]{const size_t i=oi++%count;api_ok(strix_tp_dense_plan_submit(i?pool_old0[i-1]->handle:old0plan.handle));api_ok(strix_tp_dense_plan_submit(i?pool_old1[i-1]->handle:old1plan.handle));activation();};
        auto candidate=[&]{api_ok(strix_tp_dense_pair_plan_submit(pool_new[ci++%count]));};
        baseline();const auto g=y0.values(),u=y1.values(),h=swiglu?hidden.values():std::vector<float>{};candidate();
        if(swiglu)exact(hidden.values(),h,"Benchmark SwiGLU bytes differ from legacy+stable activation");
        else {exact(y0.values(),g,"Benchmark paired q_a bytes differ");exact(y1.values(),u,"Benchmark paired kv bytes differ");}
        const std::string label=std::string("kind=pair site=")+(swiglu?"swiglu":"q_a_kv")+" bits="+std::to_string(f0.bits)+" rows="+std::to_string(rows)+" packed_bytes="+std::to_string(packed)+" pool="+std::to_string(count);
        std::printf("TP_PLAN %s strategy=splitk-pair S=%s grid=288 row_blocks=1 threads=256 launches=1 workspace=%llu\n",label.c_str(),swiglu?"16/16":"16/32",static_cast<unsigned long long>(bytes));
        paired_measure(rt,label,baseline,candidate,original,cold?int(count):repetitions,cold?"resident_disjoint_pool":"hot_fixed_addresses");
        for(auto next:pool_new)api_ok(strix_tp_dense_pair_plan_destroy(next));
        pool_base0.clear();pool_base1.clear();pool_old0.clear();pool_old1.clear();clones0.clear();clones1.clear();
        ws.canaries();bws.canaries();input.unchanged();
    }
}
} // namespace
int main(int argc,char** argv) {
    try {
        setvbuf(stdout,nullptr,_IOLBF,0);
        require(argc>=3,"usage: tp_splitk_probe identity|single|pairs|hot|cold FIXTURES [REPETITIONS]");
        const std::string suite=argv[1],dir=argv[2];Runtime rt;
        if(suite=="dot-calibrate")calibrate_dot(rt,dir);
        else if(suite=="identity" || suite=="single" || suite=="native-single") {
            if(suite=="single"){dot_gate(rt,dir);core_gate(rt,dir);}
            if(suite=="native-single")std::puts("TP_CPU_DOT_MODEL_PENDING benchmarking_forbidden=1");
            for(const char* site:{"q_b","o_b","shared_gate","shared_up","shared_down","q_a","kv"})for(unsigned bits:{4u,5u}) {
                Fixture f(dir+"/dense-"+site+"-k"+std::to_string(bits)+".strank");
                if(suite=="single")dense_edge_case(rt,f,site);
                split_case(rt,f,site,suite=="identity");
            }
        } else if(suite=="pairs") {
            for(bool swiglu:{false,true})for(bool equal:{false,true})for(unsigned b0:{4u,5u})for(unsigned b1:{4u,5u}) {
                const std::string prefix=dir+"/pair-"+std::to_string(int(equal))+"-";
                Fixture f0(prefix+(swiglu?"gate":"q_a")+"-k"+std::to_string(b0)+".strank");
                Fixture f1(prefix+(swiglu?"up":"kv")+"-k"+std::to_string(b1)+".strank");
                pair_case(rt,f0,f1,swiglu,equal);
            }
        } else if(suite=="hot" || suite=="cold") {
            const int repetitions=argc>3?std::stoi(argv[3]):4;
            require(repetitions>=1 && repetitions<=64,"Invalid benchmark repetitions");
            for(const char* site:{"q_b","o_b","shared_gate","shared_up","shared_down","q_a","kv"})for(unsigned bits:{4u,5u}) {
                Fixture f(dir+"/dense-"+site+"-k"+std::to_string(bits)+".strank");bench_dense(rt,f,site,repetitions,suite=="cold");
            }
            for(bool swiglu:{false,true})for(unsigned bits:{4u,5u}) {
                const std::string prefix=dir+"/pair-0-";
                Fixture f0(prefix+(swiglu?"gate":"q_a")+"-k"+std::to_string(bits)+".strank");
                Fixture f1(prefix+(swiglu?"up":"kv")+"-k"+std::to_string(bits)+".strank");bench_pair(rt,f0,f1,swiglu,repetitions,suite=="cold");
            }
        } else throw std::runtime_error("Unknown split-K suite");
        std::printf("PASS tp_splitk_%s\n",suite.c_str());return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL tp_splitk: %s\n",e.what());return 1;}
}
