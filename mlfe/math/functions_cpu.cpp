#include <algorithm>
#include <chrono>
#include <random>
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
void ReluFunction<double, CPUContext>(
    const int size,
    const double *x,
    double *y
    ) {
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

template <>
void ReluGradientFunction<double, CPUContext>(
    const int size,
    const double *y,
    const double *dy,
    double *dx
    ) {
    for (int i = 0; i < size; ++i) {
        dx[i] = y[i] > 0 ? dy[i] : 0;
    }
}

unsigned int GetRandomSeed(){
    int out;
    uint64_t seed = std::chrono::high_resolution_clock::
    now().time_since_epoch().count();
    std::seed_seq seeder{uint32_t(seed),uint32_t(seed >> 32)};
    ++seed;
    seeder.generate(&out, &out + 1);
    return out;
}

} /* namespace math */
} /* namespace mlfe */
