#include "mlfe/optimizers_v2/impl/cuda/kernel/adam.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace cuda_kernel{

template <typename T>
__global__ void adam_kernel(const int size,
                            T *w,
                            const T *dw,
                            T *m_hist,
                            T *v_hist,
                            const T lr,
                            const T beta1,
                            const T beta2,
                            const T eps
                           )
{
    T correction = lr * sqrt(T(1) - beta2) / (T(1) - beta1);
    CUDA_1D_KERNEL_LOOP(n, size){
        T g = dw[n];
        T mh = beta1 * m_hist[n] + (T(1) - beta1) * g;
        T vh = beta2 * v_hist[n] + (T(1) - beta2) * g * g;
        m_hist[n] = mh;
        v_hist[n] = vh;
        w[n] -= correction * mh / (sqrt(vh) + eps);
    }
}

template <>
void adam<float>(const int size,
    float *w, const float *dw, float *m_hist, float *v_hist,
    const float lr, const float beta1, const float beta2, const float eps
    )
{
    adam_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(
            size, w, dw, m_hist, v_hist,
            lr, beta1, beta2, eps
            );
}

} // namespace cuda_kernel
} // namespace mlfe