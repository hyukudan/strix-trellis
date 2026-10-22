// Private, allocation-factory validation. Does not certify arbitrary HIP pointers.
#pragma once
#include "descriptor.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
namespace tp_decode {
enum class Variant { Old, Canonical, A, B, C, D, DiagnosticD };
enum class Error { Ok, Header, Enum, Shape, Splits, Rows, Stride, Overflow,
                   Alignment, DeclaredExtent, OwnedExtent, Alias, Resource };
struct Region { uintptr_t address; uint64_t bytes; };
struct Layout { int input, output, rows; uint64_t xs, ys, a_bytes, workspace; };
// C descriptors can contain arbitrary enum object representations. Reading an
// invalid unscoped enum as a C++ enum is UB before a comparison can reject it.
template<class E> inline uint32_t enum_bits(const E& value){
    static_assert(sizeof(E)==sizeof(uint32_t),"private C enum ABI width");
    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));return bits;
}
inline bool add(uint64_t a,uint64_t b,uint64_t& out) {
    if(b>UINT64_MAX-a)return false;out=a+b;return true;
}
inline bool mul(uint64_t a,uint64_t b,uint64_t& out) {
    if(a && b>UINT64_MAX/a)return false;out=a*b;return true;
}
template<size_t N> inline bool zero(const uint64_t (&v)[N]) {
    for(auto x:v)if(x)return false;return true;
}
inline Error extent(const strix_tensor_desc& t,int width,int rows,uint64_t& bytes,uint64_t& stride) {
    if(t.struct_size!=sizeof(t)||t.abi_version!=1||!zero(t.reserved))return Error::Header;
    if(enum_bits(t.dtype)!=uint32_t(STRIX_DTYPE_F32)||t.rank<1||t.rank>2)return Error::Shape;
    if(t.rank==1) {
        if(rows!=1||t.shape[0]!=uint64_t(width)||t.shape[1]!=0||t.strides[0]!=1||t.strides[1]!=0)return Error::Stride;
        stride=width;
    } else {
        if(t.shape[0]!=uint64_t(rows)||t.shape[1]!=uint64_t(width))return Error::Shape;
        if(t.strides[1]!=1||t.strides[0]<uint64_t(width))return Error::Stride;
        stride=t.strides[0];
    }
    uint64_t elements;
    if(!mul(rows-1,stride,elements)||!add(elements,width,elements)||!mul(elements,4,bytes))return Error::Overflow;
    uint64_t end;
    if(!add(uintptr_t(t.data),t.buffer_bytes,end)||bytes>SIZE_MAX)return Error::Overflow;
    if(!t.data||uintptr_t(t.data)%4)return Error::Alignment;
    if(bytes>t.buffer_bytes)return Error::DeclaredExtent;
    return Error::Ok;
}
inline bool primary(int i,int n){return (i==1280&&n==16384)||(i==4096&&n==5120);}
inline bool eligible(Variant v,int i,int n,int r) {
    if(unsigned(v)>unsigned(Variant::DiagnosticD))return false;
    if(!primary(i,n)||r<1||r>8)return false;
    return (v!=Variant::D&&v!=Variant::DiagnosticD)||r==1||(i==4096&&r==6);
}
// Extents are immutable factory records, never supplied by the untrusted descriptor.
inline Error owned(Region request,Region owner) {
    uint64_t re,oe;
    if(!add(request.address,request.bytes,re)||!add(owner.address,owner.bytes,oe))return Error::Overflow;
    if(request.address<owner.address||re>oe)return Error::OwnedExtent;
    return Error::Ok;
}
inline Error validate(const tp_desc& d,int i,int n,Variant v,const Region (&owners)[4],Layout& l) {
    l={};
    if(d.struct_size!=sizeof(d)||d.abi_version!=1||!zero(d.reserved)||
       d.linear.struct_size!=sizeof(d.linear)||d.linear.abi_version!=1||!zero(d.linear.reserved))return Error::Header;
    if(enum_bits(d.numerics)!=uint32_t(TP_LOGICAL_K16_V1))return Error::Enum;
    if(d.logical_splits!=4)return Error::Splits;
    const auto& m=d.linear;
    if(!primary(i,n)||m.weights||!std::isfinite(m.alpha)||!std::isfinite(m.beta)||m.input.rank!=m.output.rank)return Error::Shape;
    uint64_t r=m.input.rank==1?1:m.input.shape[0];
    if(r<1||r>8)return Error::Rows;
    if(!eligible(v,i,n,int(r)))return Error::Shape;
    Layout result{i,n,int(r),0,0,0,0};uint64_t xb,yb;
    Error err=extent(m.input,i,int(r),xb,result.xs);if(err!=Error::Ok)return err;
    err=extent(m.output,n,int(r),yb,result.ys);if(err!=Error::Ok)return err;
    result.a_bytes=(2*r*i+255)&~uint64_t(255);
    result.workspace=result.a_bytes+16*r*n;
    if(!m.workspace||uintptr_t(m.workspace)%256)return Error::Alignment;
    uint64_t end;if(!add(uintptr_t(m.workspace),m.workspace_bytes,end))return Error::Overflow;
    if(m.workspace_bytes<result.workspace)return Error::DeclaredExtent;
    // Whole declared capacity is checked, including gaps and excess capacity.
    Region regions[4]={{uintptr_t(m.input.data),m.input.buffer_bytes},
        {uintptr_t(m.output.data),m.output.buffer_bytes},{uintptr_t(m.workspace),m.workspace_bytes},owners[3]};
    for(int j=0;j<4;++j){err=owned(regions[j],owners[j]);if(err!=Error::Ok)return err;}
    for(int a=0;a<4;++a)for(int b=a+1;b<4;++b) {
        uint64_t ae,be;
        if(!add(regions[a].address,regions[a].bytes,ae)||!add(regions[b].address,regions[b].bytes,be))return Error::Overflow;
        if(regions[a].address<be&&regions[b].address<ae)return Error::Alias;
    }
    l=result;return Error::Ok;
}
inline thread_local unsigned submit_depth=0;
inline size_t forbidden_calls[6]{};
inline bool forbidden(unsigned kind){if(!submit_depth)return false;++forbidden_calls[kind];return true;}
struct SubmitScope {SubmitScope(){++submit_depth;}~SubmitScope(){--submit_depth;}};
inline bool clean_submit(){for(auto x:forbidden_calls)if(x)return false;return true;}
} // namespace tp_decode
