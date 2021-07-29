#include "mlfe/optimizers/impl/cuda/kernel/adadelta.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace cuda_kernel{

template <typename T>
__global__ void adadelta_kernel(const int size,
                                T *w,
                                const T *dw,
                                T *grad_hist,
                                T *acc_hist,
                                const T lr,
                                const T momentum,
                                const T eps
                               )
{
    CUDA_1D_KERNEL_LOOP(n, size){
        T g = dw[n];
        T gh = momentum * grad_hist[n] + (T(1) - momentum) * g * g;
        grad_hist[n] = gh;
        g = sqrt((acc_hist[n] + eps) / (gh + eps)) * g;
        acc_hist[n] = momentum * acc_hist[n] + (T(1) - momentum) * g * g;
        w[n] -= lr * g;
    }
}

template <>
void adadelta<float>(const int size,
                     float *w,
                     const float *dw,
                     float *grad_hist,
                     float *acc_hist,
                     const float lr,
                     const float momentum,
                     const float eps)
{
    adadelta_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(
            size, w, dw, grad_hist, acc_hist,
            lr, momentum, eps
            );
}

} // namespace cuda_kernel
} // namespace mlfe