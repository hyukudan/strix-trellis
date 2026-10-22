#include "negative.hpp"
// Host-only supplementary mixed G1/Direct coverage; no GPU kernel definitions.
static void q1(Runtime& r,const fs::path& inputs,const fs::path& baseline){
    Fixture f((inputs/"2.stgpu").string()), h((inputs/"0.stgpu").string());
    require(f.i==1280 && f.n==32768 && f.bits==5 && f.cb==2 && f.rows==8,"Actual Q1 G1 fixture");
    require(h.i==8192 && h.n==5120 && h.bits==5 && h.cb==2,"Actual Q1 Direct fixture");
    auto ref=oracle(inputs/"2.f64"), other=oracle(inputs/"0.f64");
    setbias(f,false);setbias(h,false);
    qa_weight_reserve=f.parts.at("trellis").size()+h.parts.at("trellis").size()+2ull*(f.i+f.n+h.i+h.n)+4ull*(f.n+h.n)+2048;
    Weight w(r,f),v(r,h);no_queries();
    require(w.w!=v.w && w.w->allocation!=v.w->allocation,"Distinct real prepared owners");
    const bool old=w.w->device.fused_output_capable, fresh=v.w->device.fused_both_capable;
    require(old && fresh && w.w->device.fused_both_capable && v.w->device.fused_output_capable,"Actual observed capabilities required");
    auto wi=get(r,w.w->allocation,w.w->allocation_bytes),vi=get(r,v.w->allocation,v.w->allocation_bytes);
    Case a(r,f,w.w,8),b(r,h,v.w,1);negative_native(r,b);
    Guarded shared(std::max(a.ws.bytes,b.ws.bytes),r);
    strix_matmul_desc ds[]={a.d,b.d};for(auto& d:ds){d.workspace=nullptr;d.workspace_bytes=0;}
    strix_grouped_moe_desc g{};g.struct_size=sizeof(g);g.abi_version=1;g.groups=ds;g.group_count=2;g.workspace=shared.p;g.workspace_bytes=shared.bytes;
    uint64_t need=0;api_check(strix_grouped_moe_workspace_bytes(r.context,&g,&need));require(need==shared.bytes,"Public MAX shared workspace");
    for(int mask=0;mask<4;++mask){
        // Restore only values actually observed from prepare, never fabricate a certificate.
        w.w->device.fused_output_capable=(mask&1)?false:old;
        v.w->device.fused_both_capable=(mask&2)?false:fresh;
        a.reset(1,0);b.reset(1,0);
        Graph ga,gb;ga.capture(r,(mask&1)?3:2,[&]{a.submit();});gb.capture(r,(mask&2)?2:1,[&]{b.submit();});
        require(ga.funcs[1]==((mask&1)?ga.funcs[1]:functions[2]),"G1 must use certified old image");
        if(!(mask&2))require(gb.funcs[0]==functions[4],"Direct must use certified new image");
        ga.replay(r.stream);gb.replay(r.stream);a.oracle(ref);b.oracle(other);if(mask&1)full_partial_oracle(a,inputs,2,false);if(mask&2)full_partial_oracle(b,inputs,0,true);if(!(mask&2))exact_producer(b);
        auto ay=get(r,a.y.p,a.y.bytes),by=get(r,b.y.p,b.y.bytes),aw=get(r,a.ws.p,a.ws.bytes),bw=get(r,b.ws.p,b.ws.bytes);
        if(!(mask&1)){
            require(ay==readbytes(baseline/"q1-y"),"Unchanged G1 full Y exact V14b producer");
            require(aw==readbytes(baseline/"q1-ws"),"Unchanged G1 whole standalone workspace exact V14b");
        }
        // Derive sequential writes: G1 writes A, fallback writes A/P; R1 fallback
        // writes P only, which can overlap the preceding operator's A/P.
        std::vector<unsigned char> expected(shared.bytes,255);
        const size_t written_a=(mask&1)?a.ws.bytes:a.aligned;
        std::copy(aw.begin(),aw.begin()+written_a,expected.begin());
        if(mask&2){require(poison(std::vector<unsigned char>(bw.begin(),bw.begin()+b.aligned)),"R1 fallback A unused");finite32(bw.data()+b.aligned,bw.size()-b.aligned);std::copy(bw.begin()+b.aligned,bw.end(),expected.begin()+b.aligned);}
        else require(poison(bw),"Standalone Direct strict whole A/P poison");
        auto want=ga.funcs;want.insert(want.end(),gb.funcs.begin(),gb.funcs.end());
        for(int repeat=0;repeat<2;++repeat){
            a.reset(1,0);b.reset(1,0);hip_check(hipMemsetAsync(shared.p,255,shared.bytes,r.stream));
            Graph mixed;mixed.capture(r,3+bool(mask&1)+bool(mask&2),[&]{api_check(strix_grouped_moe(r.context,&g,&r.s));});
            require(mixed.funcs==want,"Exact mixed node functions and order versus standalone routes");
            mixed.replay(r.stream);
            require(get(r,a.y.p,a.y.bytes)==ay && get(r,b.y.p,b.y.bytes)==by,"Both full mixed Y exact standalone producers");
            a.oracle(ref);b.oracle(other);require(get(r,shared.p,shared.bytes)==expected,"Sequential shared workspace written regions");
            // Replay same graph after ordered reset, preserving owners and buffers.
            a.reset(1,0);b.reset(1,0);hip_check(hipMemsetAsync(shared.p,255,shared.bytes,r.stream));mixed.replay(r.stream);
            require(get(r,a.y.p,a.y.bytes)==ay && get(r,b.y.p,b.y.bytes)==by && get(r,shared.p,shared.bytes)==expected,"Replay full outputs/workspace");
            a.intact();b.intact();shared.check(r);no_queries();
        }
        for(int mode=0;mode<2;++mode){
            a.reset(1,0);b.reset(1,0);hip_check(hipMemsetAsync(shared.p,255,shared.bytes,r.stream));auto saved=ds[1];
            if(mode==0)ds[1].output.reserved[0]=1;else ds[1].output.buffer_bytes=4;
            Graph rejected;rejected.capture(r,0,[&]{require(strix_grouped_moe(r.context,&g,&r.s)==(mode==0?STRIX_ERROR_INVALID_ARGUMENT:STRIX_ERROR_INSUFFICIENT_BUFFER),"Mixed last-group exact rejection");});ds[1]=saved;
            require(poison(get(r,a.y.p,a.y.bytes)) && poison(get(r,b.y.p,b.y.bytes)) && poison(get(r,shared.p,shared.bytes)),"Mixed rejection zero writes");
        }
        require(get(r,w.w->allocation,w.w->allocation_bytes)==wi && get(r,v.w->allocation,v.w->allocation_bytes)==vi,"Both mixed sources immutable");
        std::cout<<"PASS Q1 mask="<<mask<<" old="<<!(mask&1)<<" new="<<!(mask&2)<<" nodes="<<want.size()<<" exact_functions=1 full_Y=1\n";
    }
    w.w->device.fused_output_capable=old;v.w->device.fused_both_capable=fresh;
    a.intact();b.intact();shared.check(r);w.close();v.close();qa_weight_reserve=0;host_budget();
}
