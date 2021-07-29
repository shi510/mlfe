#include "mlfe/operators/impl/cuda/kernel/maxpool2d.h"
#include "mlfe/operators/utils.h"
#include "mlfe/device_context/cuda_context.h"
#include <third_party/cub/cub/block/block_reduce.cuh>

namespace mlfe{
namespace cuda_kernel{

template <typename T> __global__
void maxpool2d_nhwc_kernel(
    const int B,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const T* x_ptr,
    T* y_ptr
    )
{
    CUDA_1D_KERNEL_LOOP(n, B){
        for(int ph = 0; ph < OH; ++ph){
            for(int pw = 0; pw < OW; ++pw){
                for(int c = 0; c < IC; ++c){
                    int hstart = ph * stride;
                    int wstart = pw * stride;
                    int hend = min(hstart + ksize, IH);
                    int wend = min(wstart + ksize, IW);
                    const int pool_index = ph * OW * IC + pw * IC + c;
                    T max_val = -FLT_MAX;
                    for(int h = hstart; h < hend; ++h){
                        for(int w = wstart; w < wend; ++w){
                            T cur_val = x_ptr[h * IW * IC + w * IC + c];
                            if(cur_val > max_val){
                                y_ptr[pool_index] = cur_val;
                                max_val = cur_val;
                            }
                        }
                    }
                }
            }
        }
        x_ptr += IH * IW * IC;
        y_ptr += OH * OW * IC;
    }
}

template <>
void maxpool2d_nhwc<float>(
    const int B,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const float* x_ptr,
    float* y_ptr)
{
    maxpool2d_nhwc_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(B),
        CUDA_CONTEXT_NUM_THREADS>>>(B, IC, IH, IW, OH, OW, ksize, stride, x_ptr, y_ptr);
}

template <typename T> __global__
void maxpool2d_grad_nhwc_kernel(
    const int B,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const T* x_ptr,
    const T* y_ptr,
    const T* dy_ptr,
    T* dx_ptr
    )
{
    CUDA_1D_KERNEL_LOOP(n, B){
        for(int ph = 0; ph < OH; ++ph){
            for(int pw = 0; pw < OW; ++pw){
                for(int c = 0; c < IC; ++c){
                    int hstart = ph * stride;
                    int wstart = pw * stride;
                    int hend = min(hstart + ksize, IH);
                    int wend = min(wstart + ksize, IW);
                    const int pool_index = ph * OW * IC + pw * IC + c;
                    int max_idx = -1;
                    T max_val = -FLT_MAX;
                    for(int h = hstart; h < hend; ++h){
                        for(int w = wstart; w < wend; ++w){
                            const int index = h * IW * IC + w * IC + c;
                            if(x_ptr[index] > max_val){
                                max_val = x_ptr[index];
                                max_idx = index;
                            }
                        }
                    }
                    dx_ptr[max_idx] = dy_ptr[pool_index];
                }
            }
        }
        x_ptr += IH * IW * IC;
        dy_ptr += OH * OW * IC;
        dx_ptr += IH * IW * IC;
    }
}

template <>
void maxpool2d_grad_nhwc<float>(
    const int B,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const float* x_ptr,
    const float* y_ptr,
    const float* dy_ptr,
    float* dx_ptr)
{
    maxpool2d_grad_nhwc_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(B),
        CUDA_CONTEXT_NUM_THREADS>>>(B, IC, IH, IW, OH, OW, ksize, stride, x_ptr, y_ptr, dy_ptr, dx_ptr);
}

} // namespace cuda_kernel
} // namespace mlfe
