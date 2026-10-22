// The same prohibition guard is exercised by CPU mocks and native link wrappers.
#pragma once
#include <cstddef>
namespace grouped_k3_perf {
enum class SideEffect {allocation,copy,sync,query,event,heap};
inline thread_local unsigned submit_depth=0;
inline size_t forbidden_calls[6]{};
inline bool forbidden(SideEffect kind){if(!submit_depth)return false;++forbidden_calls[unsigned(kind)];return true;}
struct SubmitScope {SubmitScope(){++submit_depth;}~SubmitScope(){--submit_depth;}};
inline bool clean_submit(){for(auto n:forbidden_calls)if(n)return false;return true;}
}
