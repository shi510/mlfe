#include "mlfe/operators_v2/maxpool2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cpu_context.h"
#include <cfloat>

namespace mlfe{
namespace operators_v2{
namespace {

template <typename T>
struct maxpool2d{
    static void run(
                    Tensor x,
                    Tensor y,
                    std::vector<int32_t> psize,
                    std::vector<int32_t> strides
                    )
    {
        int batch = x.shape()[0];
        int in_c = x.shape()[1];
        int in_h = x.shape()[2];
        int in_w = x.shape()[3];
        int out_h = y.shape()[2];
        int out_w = y.shape()[3];
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();

        for(int n = 0; n < batch; ++n){
            for(int c = 0; c < in_c; ++c){
                for(int ph = 0; ph < out_h; ++ph){
                    for(int pw = 0; pw < out_w; ++pw){
                        int hstart = ph * strides[0];
                        int wstart = pw * strides[1];
                        int hend = std::min<int>(hstart + psize[0], in_h);
                        int wend = std::min<int>(wstart + psize[1], in_w);
                        const int pool_index = ph * out_w + pw;
                        T max_val = -FLT_MAX;
                        for(int h = hstart; h < hend; ++h){
                            for(int w = wstart; w < wend; ++w){
                                T cur_val = x_ptr[h * in_w + w];
                                if(cur_val > max_val){
                                    y_ptr[pool_index] = cur_val;
                                    max_val = cur_val;
                                }
                            }
                        }
                    }
                }
                x_ptr += in_h * in_w;
                y_ptr += out_h * out_w;
            }
        }
    }
};

template <class T>
struct maxpool2d_grad
{
    static void run(
                    Tensor x,
                    Tensor dy,
                    Tensor dx,
                    std::vector<int32_t> psize,
                    std::vector<int32_t> strides
                    )
    {
        int batch;
        int in_c, in_h, in_w;
        int out_h, out_w;
        batch = x.shape()[0];
        in_c = x.shape()[1];
        in_h = x.shape()[2];
        in_w = x.shape()[3];
        out_h = dy.shape()[2];
        out_w = dy.shape()[3];
        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        math::set<T, CPUContext>(dx.size(), T(0), dx_ptr);

        for(int n = 0; n < batch; ++n){
            for(int c = 0; c < in_c; ++c){
                for(int ph = 0; ph < out_h; ++ph){
                    for(int pw = 0; pw < out_w; ++pw){
                        int hstart = ph * strides[0];
                        int wstart = pw * strides[1];
                        int hend = std::min<int>(hstart + psize[0], in_h);
                        int wend = std::min<int>(wstart + psize[1], in_w);
                        const int pool_index = ph * out_w + pw;
                        int max_idx = -1;
                        T max_val = -FLT_MAX;
                        for(int h = hstart; h < hend; ++h){
                            for(int w = wstart; w < wend; ++w){
                                const int index = h * in_w + w;
                                if(x_ptr[index] > max_val){
                                    max_val = x_ptr[index];
                                    max_idx = index;
                                }
                            }
                        }
                        dx_ptr[max_idx] = dy_ptr[pool_index];
                    }
                }
                x_ptr += in_h * in_w;
                dy_ptr += out_h * out_w;
                dx_ptr += in_h * in_w;
            }
        }
    }
};


template <typename T>
void maxpool2d_fwd_impl(
    Tensor x,
    Tensor y,
    std::vector<int32_t> psize,
    std::vector<int32_t> strides
    )
{
    maxpool2d<T>::run(x, y, psize, strides);
}

template <typename T>
void maxpool2d_bwd_impl(
    Tensor x,
    Tensor dy,
    Tensor dx,
    std::vector<int32_t> psize,
    std::vector<int32_t> strides
    )
{
    maxpool2d_grad<T>::run(x, dy, dx, psize, strides);
}

} // namespace anonymous

REGIST_OP_KERNEL(
    maxpool2d_fwd,
    maxpool2d_fwd_fn_t,
    maxpool2d_fwd_impl<float>
    );

REGIST_OP_KERNEL(
    maxpool2d_bwd,
    maxpool2d_bwd_fn_t,
    maxpool2d_bwd_impl<float>
    );

} // namespace op_kernels_cpu
} // namespace mlfe
