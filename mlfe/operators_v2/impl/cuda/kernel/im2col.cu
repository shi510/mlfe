#include "mlfe/operators_v2/impl/cuda/kernel/im2col.h"
#include "mlfe/operators_v2/utils.h"
#include "mlfe/device_context/cuda_context.h"
#include <third_party/cub/cub/block/block_reduce.cuh>

namespace mlfe{
namespace cuda_kernel{

template <typename T> __global__
void im2col_nhwc_kernel(
    const int IC,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int KH,
    const int KW,
    const int stride,
    const int padding,
    const T *im_ptr,
    T *col_ptr)
{
    const int COL_SIZE = IC * KW * KH;

    CUDA_1D_KERNEL_LOOP(c, COL_SIZE){
        int h_offset = c / IC / KW;
        int w_offset = (c / IC) % KW;
        int c_im = c % IC;
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                int im_row = h_offset + h * stride - padding;
                int im_col = w_offset + w * stride - padding;
                if(im_row >= 0 && im_col >= 0 && im_row < IH && im_col < IW){
                    int col_index = (c * OH + h) * OW + w;
                    col_ptr[col_index] = im_ptr[c_im + IC * (im_col + IW * im_row)];
                }
            }
        }
    }
}

template <>
void im2col_nhwc<float>(
    const int IC,
    const int IH,
    const int IW,
    const int OW,
    const int OH,
    const int KH,
    const int KW,
    const int stride,
    const int padding,
    const float *im_ptr,
    float *col_ptr)
{
    im2col_nhwc_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(IC * KW * KH),
        CUDA_CONTEXT_NUM_THREADS>>>(IC, IH, IW, OH, OW, KH, KW, stride, padding, im_ptr, col_ptr);
}

} // namespace cuda_kernel
} // namespace mlfe
