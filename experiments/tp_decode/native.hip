// Future ROOT-only QA and complete-operator benchmark. CPU preparation NEVER runs this file.
#include "prepared.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sys/resource.h>
using namespace tp_decode;
using Clock=std::chrono::steady_clock;
struct Fixture {
    int i,n;std::vector<uint32_t> packed;std::vector<uint16_t> u,v,a;
    std::vector<float> bias,x,yref;std::vector<double> p,mass;
    template<class T> static std::vector<T> read(std::ifstream& f,size_t count){std::vector<T> result(count);f.read(reinterpret_cast<char*>(result.data()),count*sizeof(T));require(bool(f),"truncated derivative");return result;}
    explicit Fixture(const std::string& path){std::ifstream f(path,std::ios::binary);require(bool(f),"fixture open");
        char magic[8];f.read(magic,8);require(std::memcmp(magic,"TPDECV01",8)==0,"fixture magic");
        auto header=read<uint32_t>(f,6);i=header[0];n=header[1];require(primary(i,n)&&header[2]==8&&header[3]==4&&header[4]<=1&&header[5]==5,"fixture header");
        packed=read<uint32_t>(f,13107200/4);u=read<uint16_t>(f,i);v=read<uint16_t>(f,n);bias=read<float>(f,header[4]*n);
        x=read<float>(f,8*i);a=read<uint16_t>(f,8*i);p=read<double>(f,4*8*n);mass=read<double>(f,4*8*n);yref=read<float>(f,8*n);
        require(f.peek()==EOF,"fixture trailing bytes");
    }
    std::shared_ptr<Weight> weight()const{return std::make_shared<Weight>(i,n,packed,u,v,bias);}
};
struct Snapshot {std::vector<uint16_t> a;std::vector<float> p,y;};
static std::vector<unsigned char> poison(size_t n){return std::vector<unsigned char>(n,0xa5);}
static void reset(Prepared& p,const Fixture& f,const std::vector<int>& ids,float old=0) {
    auto xb=poison(p.x->bytes()),yb=poison(p.y->bytes());
    auto xs=p.descriptor.linear.input.strides[0],ys=p.descriptor.linear.output.strides[0];
    for(size_t r=0;r<ids.size();++r){std::memcpy(xb.data()+r*xs*4,f.x.data()+ids[r]*f.i,4*f.i);
        for(int c=0;c<f.n;++c)std::memcpy(yb.data()+4*(r*ys+c),&old,4);}
    p.x->upload(xb.data(),xb.size());p.y->upload(yb.data(),yb.size());p.workspace->poison();
}
static void compare_bytes(const void* x,const void* y,size_t n,const char* label){
    if(std::memcmp(x,y,n)){auto* a=static_cast<const unsigned char*>(x);auto* b=static_cast<const unsigned char*>(y);
        size_t first=0;while(first<n&&a[first]==b[first])++first;
        std::cerr<<"FIRST_BYTE_FAILURE "<<label<<" offset="<<first<<" actual="<<unsigned(a[first])<<" expected="<<unsigned(b[first])<<'\n';
        throw std::runtime_error(label);}
}
static Snapshot inspect(Prepared& p,const Fixture& f,const std::vector<int>& ids,Variant variant,const Snapshot* expected=nullptr) {
    p.stream->wait();size_t rows=ids.size();auto xs=p.descriptor.linear.input.strides[0],ys=p.descriptor.linear.output.strides[0];
    auto xb=p.x->download<unsigned char>(p.x->bytes()),yb=p.y->download<unsigned char>(p.y->bytes());
    auto wanted=poison(xb.size());
    for(size_t r=0;r<rows;++r)std::memcpy(wanted.data()+4*r*xs,f.x.data()+ids[r]*f.i,4*f.i);
    compare_bytes(xb.data(),wanted.data(),xb.size(),"X immutability/all canaries");
    Snapshot result;result.y.resize(rows*f.n);
    for(size_t r=0;r<rows;++r){std::memcpy(result.y.data()+r*f.n,yb.data()+4*r*ys,4*f.n);
        std::fill(yb.begin()+4*r*ys,yb.begin()+4*(r*ys+f.n),0xa5);}
    wanted=poison(yb.size());compare_bytes(yb.data(),wanted.data(),yb.size(),"Y row gaps/final canaries");
    Layout l;require(p.check(p.descriptor,variant,l)==Error::Ok,"inspection descriptor");
    auto wb=p.workspace->download<unsigned char>(p.workspace->bytes());
    if(variant==Variant::D){wanted=poison(wb.size());compare_bytes(wb.data(),wanted.data(),wb.size(),"timed D global A/P must remain unused poison");}
    else {
        result.a.resize(rows*f.i);result.p.resize(4*rows*f.n);
        std::memcpy(result.a.data(),wb.data(),2*rows*f.i);std::memcpy(result.p.data(),wb.data()+l.a_bytes,16*rows*f.n);
        for(size_t j=2*rows*f.i;j<l.a_bytes;++j)require(wb[j]==0xa5,"A alignment gap");
        for(size_t j=l.workspace;j<wb.size();++j)require(wb[j]==0xa5,"workspace final canary");
        for(size_t r=0;r<rows;++r)compare_bytes(result.a.data()+r*f.i,f.a.data()+ids[r]*f.i,2*f.i,"full staged CPU A");
    }
    if(expected){
        for(size_t r=0;r<rows;++r){int row=ids[r];
            compare_bytes(result.y.data()+r*f.n,expected->y.data()+row*f.n,4*f.n,"native Y row byte invariant");
            if(variant!=Variant::D){compare_bytes(result.a.data()+r*f.i,expected->a.data()+row*f.i,2*f.i,"native A row byte invariant");
                for(int s=0;s<4;++s)compare_bytes(result.p.data()+(s*rows+r)*f.n,expected->p.data()+(s*8+row)*f.n,4*f.n,"native logical P row byte invariant");}}
    }
    require(clean_submit(),"Submit link-boundary side effect");return result;
}
template<class T> static void numerical(const float* actual,const T* ref,const double* mass,size_t n,const char* label,int split_k=0) {
    double e2=0,r2=0,maximum=0,error=0;
    for(size_t j=0;j<n;++j){double r=ref[j],a=actual[j],e=std::abs(a-r);require(std::isfinite(a)&&std::isfinite(r),"nonfinite active result");
        e2+=e*e;r2+=r*r;maximum=std::max(maximum,std::abs(r));error=std::max(error,e);
        if(mass){require(split_k>0,"split mass width");double u=2.*split_k*0x1p-24;require(e<=u/(1-u)*mass[j]+1e-30,"absolute product mass guard");}}
    double nrms=std::sqrt(e2/std::max(r2,1e-30));
    std::cout<<"NUMERIC "<<label<<" maxabs="<<error<<" refmax="<<maximum<<" NRMS="<<nrms<<'\n';
    require(error<=5e-4*std::max(1.,maximum)&&nrms<=3e-5,"STOP numerical gate");
}
static Snapshot canonical(const Fixture& f,const std::shared_ptr<Resources>& resources,const std::shared_ptr<Weight>& weight) {
    std::vector<int> ids(8);std::iota(ids.begin(),ids.end(),0);Prepared p(weight,resources,8,f.i,f.n);
    reset(p,f,ids);require(p.submit(p.descriptor,Variant::Canonical)==Error::Ok,"canonical submit");
    auto result=inspect(p,f,ids,Variant::Canonical);
    for(int s=0;s<4;++s)for(int r=0;r<8;++r)numerical(result.p.data()+(s*8+r)*f.n,f.p.data()+(s*8+r)*f.n,f.mass.data()+(s*8+r)*f.n,f.n,"P/split/row",f.i/4);
    for(int r=0;r<8;++r)numerical(result.y.data()+r*f.n,f.yref.data()+r*f.n,nullptr,f.n,"Y/row");return result;
}
static void negative_capture(Prepared& p) {
    auto original=p.descriptor;auto x=p.x->download<unsigned char>(p.x->bytes()),y=p.y->download<unsigned char>(p.y->bytes()),w=p.workspace->download<unsigned char>(p.workspace->bytes());
    std::vector<tp_desc> bad;auto d=original;d.struct_size--;bad.push_back(d);
    uint32_t raw=99;d=original;std::memcpy(&d.numerics,&raw,4);bad.push_back(d);
    raw=UINT32_MAX;d=original;std::memcpy(&d.numerics,&raw,4);bad.push_back(d);
    d=original;std::memcpy(&d.linear.input.dtype,&raw,4);bad.push_back(d);
    d=original;std::memcpy(&d.linear.output.dtype,&raw,4);bad.push_back(d);
    d=original;d.logical_splits=8;bad.push_back(d);
    d=original;d.linear.input.shape[0]=0;bad.push_back(d);d=original;d.linear.input.shape[0]=9;bad.push_back(d);
    d=original;d.linear.input.shape[1]--;bad.push_back(d);d=original;d.linear.workspace_bytes--;bad.push_back(d);
    d=original;d.linear.workspace=static_cast<char*>(d.linear.workspace)+4;bad.push_back(d);
    d=original;d.linear.input.strides[0]=UINT64_MAX;bad.push_back(d);
    d=original;d.linear.output.data=original.linear.input.data;bad.push_back(d);
    // LAST: capacity declared sufficient, actual private-owned extent exceeded.
    d=original;d.linear.workspace_bytes=p.workspace->bytes()+1;bad.push_back(d);
    for(const auto& invalid:bad){hipGraph_t graph=nullptr;hip(hipStreamBeginCapture(p.stream->get(),hipStreamCaptureModeGlobal));
        auto err=p.submit(invalid,Variant::A);hip(hipStreamEndCapture(p.stream->get(),&graph));size_t nodes=0;hip(hipGraphGetNodes(graph,nullptr,&nodes));hip(hipGraphDestroy(graph));
        require(err!=Error::Ok&&nodes==0,"invalid capture must enqueue zero nodes");}
    {hipGraph_t graph=nullptr;hip(hipStreamBeginCapture(p.stream->get(),hipStreamCaptureModeGlobal));
        auto err=p.submit(original,Variant(99));hip(hipStreamEndCapture(p.stream->get(),&graph));size_t nodes=0;hip(hipGraphGetNodes(graph,nullptr,&nodes));hip(hipGraphDestroy(graph));
        require(err!=Error::Ok&&nodes==0,"unknown private variant must enqueue zero nodes");}
    auto after=p.x->download<unsigned char>(x.size());compare_bytes(after.data(),x.data(),x.size(),"negative X immutable");
    after=p.y->download<unsigned char>(y.size());compare_bytes(after.data(),y.data(),y.size(),"negative Y immutable");
    after=p.workspace->download<unsigned char>(w.size());compare_bytes(after.data(),w.data(),w.size(),"negative workspace immutable");
}
static void old_check(const Snapshot& observed,const Fixture& f,const std::vector<int>& ids,const Snapshot& reference){
    bool a=true,p=true,y=true;size_t rows=ids.size();
    for(size_t r=0;r<rows;++r){a&=std::memcmp(observed.a.data()+r*f.i,reference.a.data()+ids[r]*f.i,2*f.i)==0;
        y&=std::memcmp(observed.y.data()+r*f.n,reference.y.data()+ids[r]*f.n,4*f.n)==0;
        numerical(observed.y.data()+r*f.n,f.yref.data()+ids[r]*f.n,nullptr,f.n,"actual V16 Y/row");
        for(int s=0;s<4;++s){p&=std::memcmp(observed.p.data()+(s*rows+r)*f.n,reference.p.data()+(s*8+ids[r])*f.n,4*f.n)==0;
            numerical(observed.p.data()+(s*rows+r)*f.n,f.p.data()+(s*8+ids[r])*f.n,f.mass.data()+(s*8+ids[r])*f.n,f.n,"actual V16 P/split/row",f.i/4);}}
    std::cout<<"OLD_BYTES A="<<a<<" P="<<p<<" Y="<<y<<" R="<<rows<<'\n';
}
static std::vector<Variant> variants(int i,int n,int r,bool old=true) {
    std::vector<Variant> result;if(old)result.push_back(Variant::Old);
    result.insert(result.end(),{Variant::Canonical,Variant::A,Variant::B,Variant::C});
    if(eligible(Variant::D,i,n,r)){result.push_back(Variant::D);result.push_back(Variant::DiagnosticD);}return result;
}
static void qa(const Fixture& f,const std::shared_ptr<Resources>& resources) {
    auto weight=f.weight();auto reference=canonical(f,resources,weight);
    auto source_weight=weight->allocation->download<unsigned char>(weight->allocation->bytes());
    for(int rows=1;rows<=8;++rows){
        // Eight cyclic windows exercise every original row in EVERY valid row position/R.
        for(auto variant:variants(f.i,f.n,rows)){
            auto p=std::make_shared<Prepared>(weight,resources,rows,f.i+(variant==Variant::Old?0:13),f.n+(variant==Variant::Old?0:19));
            for(int rotate=0;rotate<8;++rotate){std::vector<int> ids(rows);for(int r=0;r<rows;++r)ids[r]=(rotate+r)%8;
                reset(*p,f,ids);require(p->submit(p->descriptor,variant)==Error::Ok,"row window submit");
                if(variant==Variant::Old){auto observed=inspect(*p,f,ids,variant);bool exact=true;
                    for(int r=0;r<rows;++r){exact&=std::memcmp(observed.y.data()+r*f.n,reference.y.data()+ids[r]*f.n,4*f.n)==0;
                        for(int s=0;s<4;++s)numerical(observed.p.data()+(s*rows+r)*f.n,f.p.data()+(s*8+ids[r])*f.n,f.mass.data()+(s*8+ids[r])*f.n,f.n,"actual V16 P/split/row",f.i/4);
                        numerical(observed.y.data()+r*f.n,f.yref.data()+ids[r]*f.n,nullptr,f.n,"actual V16 Y/row");}
                    std::cout<<"OLD_EQUALITY R="<<rows<<" rotation="<<rotate<<" Ybytes="<<exact<<'\n';
                } else inspect(*p,f,ids,variant,&reference);
                // Every valid position retains its row while all neighbors change
                // to another independently referenced original input row.
                if(variant!=Variant::Old)for(int position=0;position<rows;++position){
                    std::vector<int> changed=ids;for(int r=0;r<rows;++r)if(r!=position)changed[r]=(ids[r]+3)%8;
                    reset(*p,f,changed);require(p->submit(p->descriptor,variant)==Error::Ok,"neighbor perturbation");
                    inspect(*p,f,changed,variant,&reference);
                }
            }
            for(bool reverse:{false,true}){std::vector<int> ids(rows);for(int r=0;r<rows;++r)ids[r]=reverse?7-r:(r*3)%8;
                reset(*p,f,ids);require(p->submit(p->descriptor,variant)==Error::Ok,"permutation submit");
                if(variant!=Variant::Old)inspect(*p,f,ids,variant,&reference);else inspect(*p,f,ids,variant);}
            std::vector<int> ids(rows);std::iota(ids.begin(),ids.end(),0);reset(*p,f,ids);
            negative_capture(*p);require(p->submit(p->descriptor,variant)==Error::Ok,"rejection retry");
            if(variant!=Variant::Old)inspect(*p,f,ids,variant,&reference);else inspect(*p,f,ids,variant);
            // Two replays, overwrite the temporary host descriptor after capture.
            reset(*p,f,ids);auto saved=p->descriptor;
            {auto alternate=std::make_shared<Prepared>(weight,resources,rows,f.i+(variant==Variant::Old?0:13),f.n+(variant==Variant::Old?0:19));
                reset(*alternate,f,ids);auto other_y=alternate->y->download<unsigned char>(alternate->y->bytes());auto other_w=alternate->workspace->download<unsigned char>(alternate->workspace->bytes());
                Graph graph(p,variant);p->descriptor=alternate->descriptor;
                for(int replay=0;replay<2;++replay){graph.replay();p->descriptor=saved;
                    if(variant!=Variant::Old)inspect(*p,f,ids,variant,&reference);else inspect(*p,f,ids,variant);
                    auto y=alternate->y->download<unsigned char>(other_y.size()),w=alternate->workspace->download<unsigned char>(other_w.size());
                    compare_bytes(y.data(),other_y.data(),y.size(),"host descriptor swap must not retarget graph Y");compare_bytes(w.data(),other_w.data(),w.size(),"host descriptor swap must not retarget graph workspace");
                    p->descriptor=alternate->descriptor;}}
            p->descriptor=saved;
        }
    }
    // Explicit batch partitions, with the same logical row producer throughout.
    for(int cut:{1,3,6})for(int part=0;part<2;++part){int first=part?cut:0,rows=part?8-cut:cut;
        std::vector<int> ids(rows);std::iota(ids.begin(),ids.end(),first);
        for(auto variant:variants(f.i,f.n,rows,false)){
            Prepared p(weight,resources,rows,f.i+3,f.n+5);reset(p,f,ids);
            require(p.submit(p.descriptor,variant)==Error::Ok,"partition submit");inspect(p,f,ids,variant,&reference);
        }
    }
    // Concurrent prepared owners on two streams; enqueue BOTH before waiting either.
    for(auto variant:{Variant::Canonical,Variant::A,Variant::B,Variant::C,Variant::D,Variant::DiagnosticD}){
        auto left=std::make_shared<Prepared>(weight,resources,1,f.i+7,f.n+11),right=std::make_shared<Prepared>(weight,resources,1,f.i+9,f.n+17);
        reset(*left,f,{2});reset(*right,f,{5});require(left->submit(left->descriptor,variant)==Error::Ok,"stream left");require(right->submit(right->descriptor,variant)==Error::Ok,"stream right");
        inspect(*left,f,{2},variant,&reference);inspect(*right,f,{5},variant,&reference);
        reset(*left,f,{2});reset(*right,f,{5});
        Graph gl(left,variant),gr(right,variant);
        for(int replay=0;replay<2;++replay){gl.replay();gr.replay();inspect(*left,f,{2},variant,&reference);inspect(*right,f,{5},variant,&reference);}
        // R1 huge row strides must not inflate the accessed/required extent.
        left->descriptor.linear.input.strides[0]=UINT64_MAX;left->descriptor.linear.output.strides[0]=UINT64_MAX;
        reset(*left,f,{2});require(left->submit(left->descriptor,variant)==Error::Ok,"R1 huge stride");inspect(*left,f,{2},variant,&reference);
        // Rank-one uses vector stride1, with the same producer.
        auto& m=left->descriptor.linear;m.input.rank=m.output.rank=1;m.input.shape[0]=f.i;m.input.shape[1]=0;m.output.shape[0]=f.n;m.output.shape[1]=0;
        m.input.strides[0]=m.output.strides[0]=1;m.input.strides[1]=m.output.strides[1]=0;
        // reset/inspect address row0 identically for either descriptor rank.
        reset(*left,f,{2});require(left->submit(left->descriptor,variant)==Error::Ok,"rank-one vector");inspect(*left,f,{2},variant,&reference);
    }
    auto after=weight->allocation->download<unsigned char>(source_weight.size());compare_bytes(after.data(),source_weight.data(),after.size(),"weights/source immutability");
    std::cout<<"PASS full native staged A/P/Y row-window permutation graph negative concurrency QA\n";
}
static float half_value(uint16_t bits){_Float16 value;std::memcpy(&value,&bits,2);return float(value);}
static std::vector<float> epilogue_reference(const Fixture& f,const std::vector<float>& bias,float alpha,float beta,float old){
    std::vector<float> y(8*f.n,0);
    for(int r=0;r<8;++r)for(int h=0;h<f.n/128;++h){float values[128]={};
        for(int s=0;s<4;++s)for(int j=0;j<128;++j)values[j]+=float(f.p[(s*8+r)*f.n+h*128+j]);
        for(int stride=1;stride<128;stride*=2)for(int j=0;j<128;++j)if(!(j&stride)){
            float a=values[j],b=values[j+stride];values[j]=a+b;values[j+stride]=a-b;}
        for(int j=0;j<128;++j){int c=h*128+j;float v=values[j]*0x1.6a09e6p-4f;v*=half_value(f.v[c]);
            if(!bias.empty())v+=bias[c];v*=alpha;if(beta!=0)v+=beta*old;y[r*f.n+c]=v;}
    }return y;
}
static void epilogue_qa(const Fixture& f,const std::shared_ptr<Resources>& resources){
    for(bool use_bias:{false,true}){std::vector<float> bias;
        if(use_bias){bias.resize(f.n);for(int c=0;c<f.n;++c)bias[c]=float((c%11)-5)*0.03125f;}
        auto weight=std::make_shared<Weight>(f.i,f.n,f.packed,f.u,f.v,bias);
        for(float alpha:{0.f,1.f,-0.7f})for(float beta:{0.f,-0.f,1.f,-0.3f}){
            std::vector<int> full(8);std::iota(full.begin(),full.end(),0);Prepared base(weight,resources,8,f.i+3,f.n+7);
            base.descriptor.linear.alpha=alpha;base.descriptor.linear.beta=beta;
            reset(base,f,full,0.25f);require(base.submit(base.descriptor,Variant::Canonical)==Error::Ok,"epilogue canonical");auto reference=inspect(base,f,full,Variant::Canonical);
            auto expected=epilogue_reference(f,bias,alpha,beta,0.25f);
            for(int r=0;r<8;++r)numerical(reference.y.data()+r*f.n,expected.data()+r*f.n,nullptr,f.n,"epilogue CPU/row");
            for(int rows=1;rows<=8;++rows)for(auto variant:variants(f.i,f.n,rows,false)){
                Prepared p(weight,resources,rows,f.i+3,f.n+7);p.descriptor.linear.alpha=alpha;p.descriptor.linear.beta=beta;
                std::vector<int> ids(rows);std::iota(ids.begin(),ids.end(),0);
                float old=beta==0?std::numeric_limits<float>::quiet_NaN():0.25f;
                reset(p,f,ids,old);require(p.submit(p.descriptor,variant)==Error::Ok,"epilogue candidate");inspect(p,f,ids,variant,&reference);
            }
        }
    }
    std::cout<<"PASS EPILOGUE bias alpha0/1/negative beta+0/-0/1/negative YoldNaN allR\n";
}
static void bench(const Fixture& f,const std::shared_ptr<Resources>& resources,const std::string& name) {
    // One geometry/pool at a time: 21 disjoint full packed copies, all traversed.
    std::vector<std::shared_ptr<Weight>> weights;for(int clone=0;clone<21;++clone)weights.push_back(f.weight());
    require(uint64_t(21)*13107200==275251200&&uint64_t(21)*13107200>=(256ull<<20),"resident packed pool");
    auto weight_bytes=poison(weights[0]->allocation->bytes());size_t offset=f.packed.size()*4;
    std::memcpy(weight_bytes.data(),f.packed.data(),offset);std::memcpy(weight_bytes.data()+offset,f.u.data(),2*f.i);offset+=2*f.i;
    std::memcpy(weight_bytes.data()+offset,f.v.data(),2*f.n);offset+=2*f.n;if(!f.bias.empty())std::memcpy(weight_bytes.data()+offset,f.bias.data(),4*f.n);
    auto reference=canonical(f,resources,weights[0]);
    for(int rows=1;rows<=8;++rows){std::vector<int> ids(rows);std::iota(ids.begin(),ids.end(),0);
        std::vector<std::shared_ptr<Prepared>> pool;auto stream=std::make_shared<Stream>();
        for(auto w:weights)pool.push_back(std::make_shared<Prepared>(w,resources,rows,f.i,f.n,stream));
        auto choices=variants(f.i,f.n,rows,false);choices.erase(std::remove(choices.begin(),choices.end(),Variant::DiagnosticD),choices.end());
        auto full_check=[&](Variant variant){for(auto& p:pool){
                if(variant==Variant::Old)old_check(inspect(*p,f,ids,variant),f,ids,reference);else inspect(*p,f,ids,variant,&reference);
            }
            if(variant==Variant::D){
                // Same body template with only owned diagnostic stores added.
                // Timed production D's poisoned globals were checked above.
                for(auto& p:pool){reset(*p,f,ids);require(p->submit(p->descriptor,Variant::DiagnosticD)==Error::Ok,"D diagnostic clone");}
                for(auto& p:pool)inspect(*p,f,ids,Variant::DiagnosticD,&reference);
            }
            for(auto& w:weights){auto actual=w->allocation->download<unsigned char>(weight_bytes.size());compare_bytes(actual.data(),weight_bytes.data(),actual.size(),"every clone packed/scales/bias/canaries immutable");}
        };
        // Full warmup and every clone before any timed sweep.
        for(auto variant:choices){for(auto& p:pool){reset(*p,f,ids);require(p->submit(p->descriptor,variant)==Error::Ok,"warmup");}full_check(variant);}
        hipEvent_t start,end;hip(hipEventCreate(&start));hip(hipEventCreate(&end));
        for(auto candidate:choices)for(int sweep=0;sweep<11;++sweep){std::vector<Variant> ordered={Variant::Old,candidate};if(sweep&1)std::reverse(ordered.begin(),ordered.end());
            for(auto variant:ordered){
                // Full per-clone gate BEFORE each sweep as well as after it.
                for(auto& p:pool){reset(*p,f,ids);require(p->submit(p->descriptor,variant)==Error::Ok,"before sweep");}full_check(variant);
                for(auto& p:pool){
                    // Reset/upload outside event/wall. beta0 means each measured op is identical.
                    reset(*p,f,ids);stream->wait();
                    auto before=Clock::now();hip(hipEventRecord(start,stream->get()));
                    require(p->submit(p->descriptor,variant)==Error::Ok,"timed submit");hip(hipEventRecord(end,stream->get()));
                    hip(hipEventSynchronize(end));auto after=Clock::now();float ms;hip(hipEventElapsedTime(&ms,start,end));
                    size_t clone=&p-&pool[0];double wall=std::chrono::duration<double,std::micro>(after-before).count();
                    std::cout<<"SAMPLE,"<<name<<','<<rows<<','<<int(candidate)<<','<<int(variant)<<','<<sweep<<','<<clone<<','<<std::setprecision(10)<<ms*1000<<','<<wall<<'\n';
                }full_check(variant); // Every A/P/Y of every clone after EACH sweep/path.
            }
        }
        // LAST all paths and all clones, including full D diagnostic A/P separately.
        for(auto variant:variants(f.i,f.n,rows)){for(auto& p:pool){reset(*p,f,ids);require(p->submit(p->descriptor,variant)==Error::Ok,"LAST submit");}full_check(variant);}
        hip(hipEventDestroy(start));hip(hipEventDestroy(end));
    }
}
int main(int argc,char** argv){try{
    require(argc==3&&std::string(argv[1])=="--root-campaign","ROOT campaign entry only");
    require(std::getenv("ROOT_GPU_LEASE")&&std::string(std::getenv("ROOT_GPU_LEASE"))=="1","ROOT lease missing");
    auto resources=std::make_shared<Resources>();std::string directory=argv[2];
    for(int layer:{0,21,39})for(const char* site:{"q_b","o_b"})for(int rank:{0,1}){
        std::string name="L"+std::to_string(layer)+"."+site+".rank"+std::to_string(rank);std::cout<<"FIXTURE "<<name<<std::endl;
        Fixture fixture(directory+"/"+name+".tp");qa(fixture,resources);epilogue_qa(fixture,resources);
    }
    std::cout<<"PASS QA_ALL12\n";
    for(const char* site:{"q_b","o_b"})for(int rank:{0,1}){std::string name=std::string("L0.")+site+".rank"+std::to_string(rank);Fixture fixture(directory+"/"+name+".tp");bench(fixture,resources,name);}
    require(live_bytes==0,"LAST allocation ownership");rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0&&uint64_t(usage.ru_maxrss)*1024<(2ull<<30),"RSS bound");
    std::cout<<"PEAK_GPU_EXPLICIT="<<peak_bytes<<" RSS="<<uint64_t(usage.ru_maxrss)*1024<<"\nPASS TP_PRIVATE_CAMPAIGN\n";return 0;
}catch(const std::exception& e){std::cerr<<"STOP "<<e.what()<<'\n';return 1;}}
