#include "optimizers.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{ namespace math{

template <>
void gradient_descent_momentum<float, CPUContext>(const int size,
                                                  float *w,
                                                  const float *dw,
                                                  float *w_momentum,
                                                  const float lr,
                                                  const float momentum,
                                                  const float decay
                                                 )
{
    for(int n = 0; n < size; ++n){
        w_momentum[n] = momentum * w_momentum[n];
        w_momentum[n] += lr * (dw[n] + decay * w[n]);
        w[n] -= w_momentum[n];
    }
}

template <>
void adadelta<float, CPUContext>(const int size,
                                 float *w,
                                 const float *dw,
                                 float *grad_hist,
                                 float *acc_hist,
                                 const float lr,
                                 const float momentum,
                                 const float eps
                                )
{
    for(int n = 0; n < size; ++n){
        float g = dw[n];
        float gh = momentum * grad_hist[n] + (1.f - momentum) * g * g;
        grad_hist[n] = gh;
        g = sqrt((acc_hist[n] + eps) / (gh + eps)) * g;
        acc_hist[n] = momentum * acc_hist[n] + (1.f - momentum) * g * g;
        w[n] -= lr * g;
    }
}

template <>
void adam<float, CPUContext>(const int size,
                             float *w,
                             const float *dw,
                             float *m_hist,
                             float *v_hist,
                             const float lr,
                             const float beta1,
                             const float beta2,
                             const float eps
                            )
{
    float correction = lr * sqrt(1.f - beta2) / (1.f - beta1);
    for(int n = 0; n < size; ++n){
        float g = dw[n];
        float mh = beta1 * m_hist[n] + (1.f - beta1) * g;
        float vh = beta2 * v_hist[n] + (1.f - beta2) * g * g;
        m_hist[n] = mh;
        v_hist[n] = vh;
        w[n] -= correction * mh / (sqrt(vh) + eps);
    }
}

} /* namespace math */
} /* namespace mlfe */
