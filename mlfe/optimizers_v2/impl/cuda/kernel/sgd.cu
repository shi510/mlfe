#include "mlfe/optimizers_v2/impl/cuda/kernel/sgd.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace cuda_kernel{

template <typename T>
__global__ void momentum_update_kernel(const int size,
                                       T *w,
                                       const T *dw,
                                       T *w_momentum,
                                       const T lr,
                                       const T momentum,
                                       const T decay
                                      )
{
    CUDA_1D_KERNEL_LOOP(n, size){
        w_momentum[n] = momentum * w_momentum[n];
        w_momentum[n] += lr * (dw[n] + decay * w[n]);
        w[n] -= w_momentum[n];
    }
}

template <>
void gradient_descent_momentum<float>(const int size,float *w,
    const float *dw, float *w_momentum, const float lr,
    const float momentum, const float decay)
{
    momentum_update_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(
            size, w, dw, w_momentum,
            lr, momentum, decay
            );
}

} // namespace cuda_kernel
} // namespace mlfe