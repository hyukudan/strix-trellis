// Test identity selection/retention/error propagation using the existing HIP
// registry. Native arithmetic qualification remains a separate GPU gate.
#define STRIX_TRELLIS_TP_MOE_A8R 1
#define STRIX_TRELLIS_TP_MOE_PIPELINED_K2 1
#define STRIX_TP_API_NO_MAIN 1
#include "test_tp_api_contract.cpp"

namespace strix_trellis {
hipError_t query_tp_moe_a8r_resources(unsigned rows, size_t shared, bool* supported) {
    return query_tp_moe_resources(rows, shared, supported);
}
hipError_t launch_tp_moe_a8r_gate_up(const TpMoeLaunch& args, hipStream_t stream) {
    ++fake::launches;tp_fake::moe=args;fake::last_stream=stream;tp_fake::order.push_back('G');return tp_fake::gate_error;
}
hipError_t launch_tp_moe_a8r_core(const TpMoeLaunch& args, hipStream_t stream) {
    ++fake::launches;tp_fake::moe=args;fake::last_stream=stream;tp_fake::order.push_back('Q');return tp_fake::core_error;
}
hipError_t launch_tp_moe_a8r_finish(const TpMoeLaunch& args, hipStream_t stream) {
    ++fake::launches;tp_fake::moe=args;fake::last_stream=stream;tp_fake::order.push_back('F');return tp_fake::finish_error;
}
hipError_t launch_tp_moe_a8r_full(const TpMoeLaunch& args, hipStream_t stream) {
    ++fake::launches;tp_fake::moe=args;fake::last_stream=stream;tp_fake::order.push_back('X');return tp_fake::gate_error;
}
}
int main() {
    {
        Environment env;
        std::vector<strix_weights> weights={prepare(env,5120,1152,2),prepare(env,5120,1152,3),
            prepare(env,1152,5120,2),prepare(env,1152,5120,3)};
        for (unsigned rows=1;rows<=8;++rows) {
            MoeCase host(rows,rows*6,weights);Buffer workspace(size_t(rows)*642048);
            host.d.workspace=workspace.pointer;host.d.workspace_bytes=workspace.bytes;
            strix_tp_moe_plan p=nullptr;
            {Counters idle;assert(strix_tp_moe_plan_create_a8r(env.context,&host.d,&env.stream,&p)==STRIX_ERROR_UNSUPPORTED_FORMAT&&!p);idle.no_work();}
            for(auto& e:host.experts){e.gate=e.up=weights[0];e.down=weights[2];}
            tp_fake::supported=false;
            {Counters idle;assert(strix_tp_moe_plan_create_a8r(env.context,&host.d,&env.stream,&p)==STRIX_ERROR_UNSUPPORTED_DEVICE&&!p);idle.no_work();}
            tp_fake::supported=true;tp_fake::resource_error=hipErrorOutOfMemory;
            {Counters idle;assert(strix_tp_moe_plan_create_a8r(env.context,&host.d,&env.stream,&p)==STRIX_ERROR_OUT_OF_MEMORY&&!p);idle.no_work();}
            tp_fake::resource_error=0;
            assert(strix_tp_moe_plan_create_a8r(env.context,&host.d,&env.stream,&p)==STRIX_SUCCESS&&p->a8r&&p->retained_count==2);
            {Counters hot;assert(strix_tp_moe_plan_submit(p)==STRIX_SUCCESS);hot.hot_path(1);assert(tp_fake::order.back()=='X');}
            assert(strix_tp_moe_plan_gate_up(p)==STRIX_SUCCESS&&tp_fake::order.back()=='G');
            assert(strix_tp_moe_plan_down_core(p)==STRIX_SUCCESS&&tp_fake::order.back()=='Q');
            assert(strix_tp_moe_plan_down_finish(p)==STRIX_SUCCESS&&tp_fake::order.back()=='F');
            {Counters idle;assert(strix_tp_moe_plan_down_reference_core(p)==STRIX_ERROR_UNSUPPORTED_FORMAT);idle.no_work();}
            tp_fake::core_error=99;
            {Counters hot;assert(strix_tp_moe_plan_down(p)==STRIX_ERROR_BACKEND);hot.hot_path(1);assert(tp_fake::order.back()=='Q');}
            tp_fake::core_error=0;
            strix_tp_moe_plan original=nullptr;
            assert(strix_tp_moe_plan_create(env.context,&host.d,&env.stream,&original)==STRIX_SUCCESS&&!original->a8r&&!original->arguments.partials);
            assert(strix_tp_moe_plan_submit(original)==STRIX_SUCCESS&&tp_fake::order.back()=='d');
            assert(strix_tp_moe_plan_destroy(original)==STRIX_SUCCESS);
            assert(strix_tp_moe_plan_destroy(p)==STRIX_SUCCESS);
            DeviceMoeCase device(rows,256,weights);device.d.workspace=workspace.pointer;device.d.workspace_bytes=workspace.bytes;
            {Counters idle;assert(strix_tp_moe_plan_create_device_a8r(env.context,&device.d,&env.stream,&p)==STRIX_ERROR_UNSUPPORTED_FORMAT&&!p);idle.no_work();}
            for(auto& e:device.experts){e.gate=e.up=weights[0];e.down=weights[2];}
            assert(strix_tp_moe_plan_create_device_a8r(env.context,&device.d,&env.stream,&p)==STRIX_SUCCESS&&p->a8r&&p->arguments.bank_count==256);
            {Counters hot;assert(strix_tp_moe_plan_submit(p)==STRIX_SUCCESS);hot.hot_path(1);assert(tp_fake::order.back()=='X');}
            assert(strix_tp_moe_plan_destroy(p)==STRIX_SUCCESS);
            for(auto w:weights)assert(w->plan_refs==0);
        }
        for(auto w:weights)assert(strix_weights_destroy(w)==STRIX_SUCCESS);
    }
    assert(fake::buffers.empty());
    std::cout<<"PASS explicit A8R selection, K3 rejection, retained weights, device bank256, error stops, original S4 isolation\n";
}
