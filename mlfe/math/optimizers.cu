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

template <typename T>
__global__ void adadelta_kernel(const int size,
                                T *w,
                                T *dw,
                                T *grad_hist,
                                T *acc_hist,
                                T lr,
                                T momentum,
                                T eps
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
void adadelta<float, CUDAContext>(const int size,
                                  float *w,
                                  float *dw,
                                  float *grad_hist,
                                  float *acc_hist,
                                  float lr,
                                  float momentum,
                                  float eps
                                 )
{
    adadelta_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(
            size, w, dw, grad_hist, acc_hist,
            lr, momentum, eps
            );
}

template <typename T>
__global__ void adam_kernel(const int size,
                            T *w,
                            T *dw,
                            T *m_hist,
                            T *v_hist,
                            T lr,
                            T beta1,
                            T beta2,
                            T eps
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
void adam<float, CUDAContext>(const int size,
                              float *w,
                              float *dw,
                              float *m_hist,
                              float *v_hist,
                              float lr,
                              float beta1,
                              float beta2,
                              float eps
                             )
{
    adam_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(
            size, w, dw, m_hist, v_hist,
            lr, beta1, beta2, eps
            );
}

} // end namespace math
} // end namespace mlfe
