#include "optimizers.h"
#include "../device_context/cpu_context.h"

namespace mlfe{ namespace math{

template <>
void gradient_descent_momentum<float, CPUContext>(const int size,
                                                  float *w,
                                                  float *dw,
                                                  float *w_momentum,
                                                  float lr,
                                                  float momentum,
                                                  float decay
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
                                 float *dw,
                                 float *grad_hist,
                                 float *acc_hist,
                                 float lr,
                                 float momentum,
                                 float eps
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
} /* namespace math */
} /* namespace mlfe */
