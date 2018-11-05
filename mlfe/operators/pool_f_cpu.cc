#include "../core/op_algo.h"
#include "../core/device.h"
#include "../math/basic_functions.h"
#include "../math/transform.h"
#include "../device_context/cpu_context.h"
#include <algorithm>
#include <cfloat>

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class MaxPool : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPool(OpAlgoContext *oac) : OpAlgo(oac, "MaxPool"){
        using IntVec = std::vector<type::int32::T>;
        y = oac->get_output(0);
        x = y.get_children()[0];
        idx = y.get_children()[1];
        filters_hw = oac->get_attr<IntVec>("kernel");
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");

        batch = x.Shape()[0];
        in_c = x.Shape()[1];
        in_h = x.Shape()[2];
        in_w = x.Shape()[3];
        out_h = y.Shape()[2];
        out_w = y.Shape()[3];
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto idx_ptr = idx.mutable_device_data<int>();
        auto y_ptr = y.mutable_device_data<T>();

        math::set<T, CPUContext>(
            y.Size(),
            T(-FLT_MAX),
            y.mutable_device_data<T>()
            );

        for(int n = 0; n < batch; ++n){
            for(int c = 0; c < in_c; ++c){
                for(int ph = 0; ph < out_h; ++ph){
                    for(int pw = 0; pw < out_w; ++pw){
                        int hstart = ph * strides[0];
                        int wstart = pw * strides[1];
                        int hend = std::min<int>(hstart + filters_hw[0], in_h);
                        int wend = std::min<int>(wstart + filters_hw[1], in_w);
                        const int pool_index = ph * out_w + pw;
                        for(int h = hstart; h < hend; ++h){
                            for(int w = wstart; w < wend; ++w){
                                const int index = h * in_w + w;
                                if(x_ptr[index] > y_ptr[pool_index]){
                                    y_ptr[pool_index] = x_ptr[index];
                                    idx_ptr[pool_index] = index;
                                }
                            }
                        }
                    }
                }
                x_ptr += in_h * in_w;
                y_ptr += out_h * out_w;
                idx_ptr += out_h * out_w;
            }
        }
    }
private:
    Tensor x;
    Tensor idx;
    Tensor y;
    int batch;
    int in_c, in_h, in_w;
    int out_h, out_w;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Output("IDX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class MaxPoolGrad : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPoolGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        idx = dx.get_children()[1];
        dy = dx.get_children()[3];
        filters_hw = oac->get_attr<IntVec>("kernel");
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");

        batch = x.Shape()[0];
        in_c = x.Shape()[1];
        in_h = x.Shape()[2];
        in_w = x.Shape()[3];
        out_h = dy.Shape()[2];
        out_w = dy.Shape()[3];
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto idx_ptr = idx.device_data<int>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        math::set<T, CPUContext>(
            dx.Size(),
            static_cast<T>(0),
            dx_ptr
            );

        for(int n = 0; n < batch; ++n){
            for(int c = 0; c < in_c; ++c){
                for(int ph = 0; ph < out_h; ++ph){
                    for(int pw = 0; pw < out_w; ++pw){
                        const int index = ph * out_w + pw;
                        const int bottom_index = idx_ptr[index];
                        dx_ptr[bottom_index] += dy_ptr[index];
                    }
                }
                dx_ptr += in_h * in_w;
                dy_ptr += out_h * out_w;
                idx_ptr += out_h * out_w;
            }
        }
    }

private:
    Tensor x;
    Tensor idx;
    Tensor dy;
    Tensor dx;
    int batch;
    int in_c, in_h, in_w;
    int out_h, out_w;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_GRAD_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Input("IDX", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
