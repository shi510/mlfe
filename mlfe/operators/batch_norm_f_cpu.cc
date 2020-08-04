#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/math/blas.h"
#include <iostream>

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class BatchNormSpatial : public OpAlgo
{
    using T = typename Tp::T;

public:
    BatchNormSpatial(OpAlgoContext *oac) : OpAlgo(oac, "BatchNormSpatial")
    {
        y = oac->get_output(0);
        x = oac->get_input(0);
        scales = oac->get_input(1);
        biases = oac->get_input(2);
        running_mean = oac->get_attr<Tensor>("running_mean");
        running_var = oac->get_attr<Tensor>("running_var");
        eps = oac->get_attr<float>("eps");
        resize();
    }

    void resize() override {
        y.resize(x.shape());
        N = y.shape()[0];
        H = y.shape()[1];
        W = y.shape()[2];
        C = y.shape()[3];
    }

    void Compute(op_algo_runtime_context& rc) override
    {
        const T* x_ptr = x.device_data<T>();
        T* y_ptr = y.mutable_device_data<T>();
        const T* mean_ptr = running_mean.device_data<T>();
        const T* var_ptr = running_var.device_data<T>();
        const T* scale_ptr = scales.device_data<T>();
        const T* b_ptr = biases.device_data<T>();
        if(rc.training()){
            std::cout<<"Not support BatchNorm training operator on CPU."<<std::endl;
            switch(get_data_order_prefer()){
                case data_order::nhwc:{
                    break;
                }
                case data_order::nchw:{
                    break;
                }
            }
        }
        switch(get_data_order_prefer()){
        case data_order::nhwc:{
            for(int m = 0; m < C; ++m){
                auto weight = scale_ptr[m] / std::sqrt(var_ptr[m] + eps);
                auto bias = b_ptr[m] - weight * mean_ptr[m];
                for(int i = 0;  i < N; ++i){
                    for(int j = 0; j < H; ++j){
                        for(int k = 0; k < W; ++k){
                            int idx = i*(H*W*C) + j*(W*C) + k*C + m;
                            y_ptr[idx] = x_ptr[idx] * weight + bias;
                        }
                    }
                }
            }
            break;
        } // end case data_order::nhwc
        case data_order::nchw:{
            for(int m = 0; m < C; ++m){
                auto weight = scale_ptr[m] / std::sqrt(var_ptr[m] + eps);
                auto bias = b_ptr[m] - weight * mean_ptr[m];
                for(int i = 0;  i < N; ++i){
                    for(int j = 0; j < H; ++j){
                        for(int k = 0; k < W; ++k){
                            int idx = i*(C*H*W) + m*(H*W) + j*W + k;
                            y_ptr[idx] = x_ptr[idx] * weight + bias;
                        }
                    }
                }
            }
            break;
        } // end case data_order::nchw
        } // end switch(get_data_order_prefer())
    }

private:
    Tensor x;
    Tensor y;
    Tensor scales;
    Tensor biases;
    Tensor running_mean;
    Tensor running_var;
    float eps;
    int N;
    int H;
    int W;
    int C;
};

REGIST_OP_ALGO(BatchNormSpatial)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo> {
        using T = BatchNormSpatial<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
