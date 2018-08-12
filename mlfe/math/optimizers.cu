#include "optimizers.h"
#include "../device_context/cuda_context.h"

namespace mlfe{ namespace math{
template <typename T>
__global__ void momentum_update_kernel(const int size,
                                       T *w,
                                       T *dw,
                                       T *w_momentum,
                                       T lr,
                                       T momentum,
                                       T decay
                                      )
{
    CUDA_1D_KERNEL_LOOP(n, size){
        w_momentum[n] = momentum * w_momentum[n];
        w_momentum[n] += lr * (dw[n] + decay * w[n]);
        w[n] -= w_momentum[n];
    }
}

template <>
void gradient_descent_momentum<float, CUDAContext>(const int size,
                                                   float *w,
                                                   float *dw,
                                                   float *w_momentum,
                                                   float lr,
                                                   float momentum,
                                                   float decay
                                                  )
{
    momentum_update_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(
            size, w, dw, w_momentum,
            lr, momentum, decay
            );
}

} // end namespace math
} // end namespace mlfe
