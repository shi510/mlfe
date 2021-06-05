#include "mlfe/operators_v2/impl/cuda/kernel/col2im.h"
#include "mlfe/operators_v2/utils.h"
#include "mlfe/device_context/cuda_context.h"
#include <third_party/cub/cub/block/block_reduce.cuh>

namespace mlfe{
namespace cuda_kernel{

template <typename T> __global__
void col2im_nhwc_kernel(
    const T* data_col,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const int pad,
    T* data_im
    )
{
    const int COL_SIZE = ksize * ksize * IC;

    CUDA_1D_KERNEL_LOOP(c, COL_SIZE){
        int h_offset = c / IC / ksize;
        int w_offset = (c / IC) % ksize;
        int c_im = c % IC;
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                if (im_row >= 0 && im_col >= 0 && im_row < IH && im_col < IW){
                    int col_index = (c * OH + h) * OW + w;
                    data_im[c_im + IC *(im_col + im_row*IW)] += data_col[col_index];
                }
                
            }
        }
    }
}

template <>
void col2im_nhwc<float>(
    const float* data_col,
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int ksize,
    const int stride,
    const int padding,
    float* data_im)
{
    col2im_nhwc_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(ksize * ksize * IC),
        CUDA_CONTEXT_NUM_THREADS>>>(data_col, IC, IH, IW, OH, OW, ksize, stride, padding, data_im);
}

} // namespace cuda_kernel
} // namespace mlfe
