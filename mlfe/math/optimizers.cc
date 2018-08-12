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

} /* namespace math */
} /* namespace mlfe */
