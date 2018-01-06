#include <algorithm>
#include "functions.hpp"
#include "../device_context/cpu_context.hpp"

namespace mlfe { namespace math {

template <>
void ReluFunction<float, CPUContext>(
                                     const int size,
                                     const float *x,
                                     float *y
                                     ){
    for (int i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

template <>
void ReluGradientFunction<float, CPUContext>(
                                             const int size,
                                             const float *y,
                                             const float *dy,
                                             float *dx
                                             ){
    for (int i = 0; i < size; ++i) {
        dx[i] = y[i] > 0 ? dy[i] : 0;
    }
}

} /* namespace math */
} /* namespace mlfe */
