#include "mlfe/core/op_algo.h"
#include "mlfe/math/activations.h"
#include "mlfe/device_context/cpu_context.h"
#include <memory>
#include <algorithm>
#include <iostream>
#include <xnnpack.h>

namespace mlfe{
namespace algorithm_xnnpack{

/*
 * Data order: NHWC or NC (Dimension should be larger than 2)
 * Input: X
 * Output: Y
 */
template <class Tp>
class ReLU : public OpAlgo{
using T = typename Tp::T;
public:
    ReLU(OpAlgoContext *oac) : OpAlgo(oac, "ReLU"){
        if(xnn_status_success != xnn_initialize(nullptr)){
            std::cout<<"xnn_status_success != xnn_initialize"<<std::endl;
            exit(1);
        }
        y = oac->get_output(0);
        x = oac->get_input(0);
        resize();
    }

    void resize() override {
        size = x.size();
        batch = x.shape()[0];
        channel = std::accumulate(x.shape().begin()+1, x.shape().end(), 0);
        w.resize(channel);
        y.resize(x.shape());
        std::fill(w.begin(), w.end(), 0.f);
        
        auto status = xnn_create_prelu_nc_f32(channel,
            channel, channel, w.data(), 0, &prelu_op);
        if(xnn_status_success != status || nullptr == prelu_op){
            std::cout<<"xnn_status_success != xnn_create_prelu_nc_f32"<<std::endl;
            exit(1);
        }
        
        status = xnn_setup_prelu_nc_f32(prelu_op,
            batch, x.device_data<T>(),
            y.mutable_device_data<T>(), nullptr);
        if(xnn_status_success != status){
            std::cout<<"xnn_status_success != xnn_setup_prelu_nc_f32"<<std::endl;
            exit(1);
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        if(xnn_status_success != xnn_run_operator(prelu_op, nullptr)){
            std::cout<<"xnn_status_success != xnn_run_operator"<<std::endl;
            exit(1);
        }
    }

    ~ReLU(){
        if(xnn_status_success != xnn_delete_operator(prelu_op)){
            std::cout<<"xnn_status_success != xnn_delete_operator"<<std::endl;
            exit(1);
        }
    }

private:
    Tensor x;
    Tensor y;
    std::vector<float> w;
    int batch;
    int channel;
    int size;
    xnn_operator_t prelu_op = nullptr;
};

REGIST_OP_ALGO(ReLU)
    .Device("CPU(XNNPACK)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLU<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace xnnpack
} // end namespace mlfe
