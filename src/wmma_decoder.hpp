// Private minimal-decoder experiment; no production API or policy changes.
#pragma once
#include "backend.hpp"
#include <memory>

namespace strix_trellis::decoder {
enum class Mode : int { baseline, aligned, saddr };
enum class Layout : int { m16_a16_plain, m64_a16_xor, m64_a18_xor };
struct Range { const void* pointer; size_t capacity; };
struct Request {
    DeviceWeights weights;
    Range packed, a, partial;
    int rows, splits, a_stride, partial_stride, total_rows;
    Layout layout;
    Mode mode;
    hipStream_t stream;
    bool diagnostic = false; // Only N16/32/48; never the measured domain.
};
class Token final {
    const Request request_;
    explicit Token(const Request& r):request_(r) {}
    friend hipError_t prepare(const Request&,std::unique_ptr<const Token>&);
    friend hipError_t dump(const Token&,uint16_t*,uint16_t*,uint16_t*,int,int,int);
public:
    Token(const Token&)=delete;
    Token& operator=(const Token&)=delete;
    hipError_t submit() const;
};
// Failure clears out, including when replacing a previously valid token.
hipError_t prepare(const Request&,std::unique_ptr<const Token>& out);
// Untimed staging probe: complete A and physical + logical Q, one CTA/K16.
hipError_t dump(const Token&,uint16_t* a,uint16_t* physical,uint16_t* logical,
    int first_row,int first_n,int tile_k);
}
