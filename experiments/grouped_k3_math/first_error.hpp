// Shared by native submit and CPU fault injection. Error values stay opaque.
#pragma once
namespace grouped_k3_math {
template<class Enqueue,class Error>
inline Error enqueue_three(Enqueue&& enqueue,Error success) {
    auto e=enqueue(0);if(e!=success)return e;
    e=enqueue(1);if(e!=success)return e;
    return enqueue(2);
}
}
