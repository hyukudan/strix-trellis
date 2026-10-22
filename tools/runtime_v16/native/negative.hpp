// Additional host-only rejection checks, inside the same five-probe context.
static void negative_native(Runtime& r,Case& c){
    struct Managed {
        void* p=nullptr;size_t bytes;
        explicit Managed(size_t n):bytes(n){qa_reserve(n);hip_check(hipMallocManaged(&p,n));}
        ~Managed(){if(p)(void)hipFree(p);qa_live-=bytes;}
    } managed(c.x.bytes);
    for(int mode=0;mode<11;++mode){
        c.reset(1,0);auto bad=c.d;auto status=STRIX_ERROR_INVALID_ARGUMENT;
        switch(mode){
        case 0:bad.input.dtype=STRIX_DTYPE_F16;status=STRIX_ERROR_UNSUPPORTED_FORMAT;break;
        case 1:bad.output.dtype=STRIX_DTYPE_BF16;status=STRIX_ERROR_UNSUPPORTED_FORMAT;break;
        case 2:bad.input.rank=1;status=STRIX_ERROR_UNSUPPORTED_FORMAT;break;
        case 3:bad.input.data=c.f.parts.at("x").data();break;
        case 4:bad.input.data=managed.p;break;
        case 5:bad.workspace_bytes=c.ws.bytes-1;status=STRIX_ERROR_INSUFFICIENT_BUFFER;break;
        case 6:bad.workspace=c.ws.p+4;break;
        case 7:bad.workspace=c.y.p;break;
        case 8:bad.input.buffer_bytes=4;status=STRIX_ERROR_INSUFFICIENT_BUFFER;break;
        case 9:bad.output.reserved[0]=1;break;
        case 10:bad.output.data=c.x.p;break;
        }
        Graph rejected;rejected.capture(r,0,[&]{auto got=strix_gemv(r.context,&bad,&r.s);require(got==status,"Native dtype/storage rejection mode="+std::to_string(mode)+" expected="+std::to_string(int(status))+" actual="+std::to_string(int(got)));(void)hipGetLastError();});
        require(poison(get(r,c.y.p,c.y.bytes)) && poison(get(r,c.ws.p,c.ws.bytes)),"Native rejection modified buffers");c.intact();no_queries();
        std::cout<<"PASS native_negative="<<mode<<" status="<<int(status)<<" nodes=0\n";
    }
}
