// CPU integration assertions. Included after generated production host adapters.
// No GPU implementation, execution, timing or numerical claim is made here.
static strix_matmul_desc make_call(strix_weights w, void* x, void* y, int rows) {
    strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights=w;
    d.input.data=x; d.output.data=y;
    for (auto* t : {&d.input,&d.output}) {
        int width=t==&d.input?w->device.input:w->device.output;
        t->rank=2; t->shape[0]=rows; t->shape[1]=width;
        t->strides[0]=width; t->strides[1]=1;
        t->buffer_bytes=size_t(rows)*width*4;
    }
    return d;
}
static void expect_nodes(std::initializer_list<const char*> expected) {
    assert(cpu_nodes.size()==expected.size());
    size_t i=0; for (auto e:expected) assert(cpu_nodes[i++]==e);
    cpu_nodes.clear();
}
int main() {
    using namespace strix_trellis;
    static_assert(sizeof(DeviceWeights)==56);
    static_assert(offsetof(DeviceWeights,fused_output_capable)==48);
    static_assert(offsetof(DeviceWeights,fused_both_capable)==49);
    assert(!DeviceWeights{}.fused_output_capable && !DeviceWeights{}.fused_both_capable);
    assert(strix_get_abi_version()==1);
    strix_stream s{}; s.struct_size=sizeof(s); s.abi_version=1;
    unsigned resource_cases=0;
    // Exercise the real resource template with independent threshold boundaries.
    for (int threads:{0,1023,1024,1025}) for (size_t stat:{0u,2047u,2048u,2049u})
    for (size_t local:{0u,1u}) for (int dyn:{-1,0,16383,16384,63488})
    for (size_t available:{18431u,18432u,65536u}) for (int error:{0,99}) {
        bool supported=true; int calls=0;
        int e=probe_fused_both_image<int,hipFuncAttributes>(available,&supported,0,3,
            reinterpret_cast<void*>(20),[&](hipFuncAttributes* a,const void* f) {
                assert(f==reinterpret_cast<void*>(20)); ++calls;
                *a={threads,stat,local,dyn}; return error;
            });
        assert(calls==1 && e==error);
        assert(supported==(!error && threads>=1024 && stat==2048 && local==0 &&
                          dyn>=16384 && available>=18432)); ++resource_cases;
    }
    // Context creation: old insufficiency cannot hide a later new query error.
    strix_context_desc cd{}; cd.struct_size=sizeof(cd); cd.abi_version=1;
    unsigned contexts=0;
    for (int active:{0,1}) for (size_t shared:{18431u,18432u,24575u,24576u,65536u})
    for (int threads:{1023,1024}) for (int dimension:{1023,1024}) {
        fake::device=active; fake::shared_bytes=shared;
        fake::max_threads=threads; fake::max_dimension=dimension;
        int old=fake::attribute_calls, fresh=fake::both_attributes;
        strix_context c=nullptr; assert(strix_context_create(&cd,&c)==STRIX_SUCCESS);
        bool viable=active==0 && threads>=1024 && dimension>=1024;
        assert(c->fused_output_capable==(viable && shared>=24576));
        assert(c->fused_both_capable==(viable && shared>=18432));
        assert(fake::attribute_calls-old==(viable && shared>=24576?4:0));
        assert(fake::both_attributes-fresh==(viable && shared>=18432?1:0));
        assert(strix_context_destroy(c)==STRIX_SUCCESS);
        ++contexts;
    }
    fake::device=0; fake::shared_bytes=65536;
    fake::max_threads=fake::max_dimension=1024;
    fake::insufficient_index=0; fake::both_error=99;
    strix_context rejected=nullptr; int before=fake::attribute_calls;
    assert(strix_context_create(&cd,&rejected)==STRIX_ERROR_BACKEND && !rejected);
    assert(fake::attribute_calls-before==4);
    fake::insufficient_index=-1; fake::both_error=0;
    strix_context c=nullptr; assert(strix_context_create(&cd,&c)==STRIX_SUCCESS);
    int old_queries=fake::attribute_calls,new_queries=fake::both_attributes;
    unsigned routes=0,negatives=0,mixed_g1=0;
    for (int input:{8192,2304}) {
        strix_exl3_metadata_v1 m{};
        m.struct_size=sizeof(m); m.transport_version=1; m.rows=5120; m.columns=input;
        m.bits_x2=10; m.codebook=STRIX_CODEBOOK_MUL1; m.marker=STRIX_EXL3_MARKER_MUL1;
        m.trellis_bytes=uint64_t(input)*5120*5/8;
        m.suh_offset=m.trellis_bytes; m.suh_bytes=input*2;
        m.svh_offset=m.suh_offset+m.suh_bytes; m.svh_bytes=5120*2;
        strix_weights w=nullptr;
        {
            std::vector<unsigned char> source(size_t(m.svh_offset+m.svh_bytes),0x35);
            strix_exl3_transport_desc t{}; t.struct_size=sizeof(t); t.abi_version=1;
            t.format=STRIX_TRANSPORT_EXL3_CANONICAL; t.transport_version=1;
            t.rows=m.rows; t.columns=m.columns; t.metadata=&m; t.metadata_bytes=sizeof(m);
            t.data=source.data(); t.data_bytes=source.size();
            assert(strix_weights_prepare(c,&t,&s,&w)==STRIX_SUCCESS);
        } // The real prepare owns its copy; the transport has expired.
        assert(w->owner==c && w->device.fused_both_capable && w->device.fused_output_capable);
        assert(*reinterpret_cast<const unsigned char*>(w->device.trellis)==0x35);
        assert(strix_context_destroy(c)==STRIX_ERROR_INVALID_ARGUMENT);
        void *x=nullptr,*y=nullptr,*x2=nullptr,*y2=nullptr,*workspace=nullptr;
        hipMalloc(&x,8*size_t(input)*4); hipMalloc(&y,8*5120*4);
        hipMalloc(&x2,8*size_t(input)*4); hipMalloc(&y2,8*5120*4);
        size_t maximum=linear_workspace_bytes(w->device,8); hipMalloc(&workspace,maximum);
        for (bool old_cap:{false,true}) for (bool new_cap:{false,true})
        for (int rows:{1,8}) for (bool prefer:{false,true}) {
            w->device.fused_output_capable=old_cap; w->device.fused_both_capable=new_cap;
            auto d=make_call(w,x,y,rows); uint64_t n=0;
            assert(strix_matmul_workspace_bytes(c,&d,&n)==STRIX_SUCCESS);
            assert(n==((size_t(rows)*input*2+255)&~size_t(255))+size_t(4)*rows*5120*4);
            d.workspace=workspace; d.workspace_bytes=n;
            assert((prefer?strix_gemm(c,&d,&s):strix_gemv(c,&d,&s))==STRIX_SUCCESS);
            if (prefer) expect_nodes({"input","wmma","output"});
            else if (rows==1 && new_cap) expect_nodes({"direct"});
            else if (rows==1) expect_nodes({"fused-input","output"});
            else if (old_cap) expect_nodes({"input","old-direct"});
            else expect_nodes({"input","lds","output"});
            ++routes;
        }
        w->device.fused_both_capable=w->device.fused_output_capable=true;
        auto d=make_call(w,x,y,1); d.workspace=workspace; d.workspace_bytes=maximum;
        cpu_fail="direct";
        assert(strix_gemv(c,&d,&s)==STRIX_ERROR_BACKEND); expect_nodes({"direct"});
        cpu_fail.clear();
        d.workspace_bytes=0;
        assert(strix_gemv(c,&d,&s)==STRIX_ERROR_INSUFFICIENT_BUFFER); expect_nodes({});
        d.workspace_bytes=maximum; w->device.fused_both_capable=false;
        cpu_fail="fused-input";
        assert(strix_gemv(c,&d,&s)==STRIX_ERROR_BACKEND); expect_nodes({"fused-input"});
        cpu_fail.clear(); w->device.fused_both_capable=true;
        strix_matmul_desc ds[]={make_call(w,x,y,8),make_call(w,x2,y2,1)};
        strix_grouped_moe_desc g{}; g.struct_size=sizeof(g); g.abi_version=1;
        g.groups=ds; g.group_count=2; g.workspace=workspace; g.workspace_bytes=maximum;
        assert(strix_grouped_moe(c,&g,&s)==STRIX_SUCCESS);
        expect_nodes({"input","old-direct","direct"});
        // A separately reserved, nonalias allocation isolates actual extent rejection.
        void* short_y=nullptr; hipMalloc(&short_y,65536);
        // Keep a full logical output apart from other host allocations, while
        // the fake driver exposes a four-byte extent. Native QA must derive
        // this isolation from actual hipMemGetAddressRange, not this model.
        fake::buffers[reinterpret_cast<uintptr_t>(short_y)]=4;
        auto saved=ds[1];
        for (int mode=0;mode<3;++mode) {
            if(mode==0) ds[1].output.reserved[0]=1;
            if(mode==1) ds[1].output.buffer_bytes=4;
            if(mode==2) ds[1].output.data=short_y;
            uint64_t n=0;
            if(mode==2) assert(strix_grouped_moe_workspace_bytes(c,&g,&n)==STRIX_SUCCESS && n==maximum);
            auto status=strix_grouped_moe(c,&g,&s);
            assert(status==(mode==0?STRIX_ERROR_INVALID_ARGUMENT:STRIX_ERROR_INSUFFICIENT_BUFFER));
            expect_nodes({}); ds[1]=saved; ++negatives;
        }
        // The frozen native harness currently uses same-shape G4 here. Model
        // the requested cross-shape G1/R1 integration independently on CPU.
        auto gm=m; gm.rows=32768; gm.columns=1280;
        gm.trellis_bytes=uint64_t(gm.rows)*gm.columns*5/8;
        gm.suh_offset=gm.trellis_bytes; gm.suh_bytes=gm.columns*2;
        gm.svh_offset=gm.suh_offset+gm.suh_bytes; gm.svh_bytes=gm.rows*2;
        strix_weights gw=nullptr;
        {
            std::vector<unsigned char> source(size_t(gm.svh_offset+gm.svh_bytes),0x37);
            strix_exl3_transport_desc t{}; t.struct_size=sizeof(t); t.abi_version=1;
            t.format=STRIX_TRANSPORT_EXL3_CANONICAL; t.transport_version=1;
            t.rows=gm.rows; t.columns=gm.columns; t.metadata=&gm; t.metadata_bytes=sizeof(gm);
            t.data=source.data(); t.data_bytes=source.size();
            assert(strix_weights_prepare(c,&t,&s,&gw)==STRIX_SUCCESS);
        }
        void *gx=nullptr,*gy=nullptr,*gws=nullptr;
        hipMalloc(&gx,8*1280*4); hipMalloc(&gy,8*32768*4);
        size_t gmaximum=linear_workspace_bytes(gw->device,8); hipMalloc(&gws,gmaximum);
        ds[0]=make_call(gw,gx,gy,8); ds[1]=make_call(w,x,y,1);
        g.workspace=gws; g.workspace_bytes=gmaximum;
        for(bool old_cap:{false,true}) for(bool new_cap:{false,true}) {
            gw->device.fused_output_capable=old_cap; gw->device.fused_both_capable=new_cap;
            w->device.fused_output_capable=old_cap; w->device.fused_both_capable=new_cap;
            assert(strix_grouped_moe(c,&g,&s)==STRIX_SUCCESS);
            if(old_cap && new_cap) expect_nodes({"input","old-direct-g1","direct"});
            else if(old_cap) expect_nodes({"input","old-direct-g1","fused-input","output"});
            else if(new_cap) expect_nodes({"input","lds","output","direct"});
            else expect_nodes({"input","lds","output","fused-input","output"});
            ++mixed_g1;
        }
        for(void* p:{gx,gy,gws}) hipFree(p);
        assert(strix_weights_destroy(gw)==STRIX_SUCCESS);
        for (void* p:{x,y,x2,y2,workspace,short_y}) hipFree(p);
        assert(strix_weights_destroy(w)==STRIX_SUCCESS);
    }
    assert(fake::attribute_calls==old_queries && fake::both_attributes==new_queries);
    assert(strix_context_destroy(c)==STRIX_SUCCESS && fake::buffers.empty());
    std::cout<<"PASS CPU host integration resource_cases="<<resource_cases
             <<" context_cases="<<contexts<<" routes="<<routes<<" isolated_last_group_negatives="<<negatives
             <<" mixed_g1_flag_cases="<<mixed_g1<<'\n';
}
