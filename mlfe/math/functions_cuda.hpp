#ifndef __MATH_FUNCTIONS_CUDA_HPP__
#define __MATH_FUNCTIONS_CUDA_HPP__
#include <curand_kernel.h>

namespace mlfe { namespace math {

void InitCurand(unsigned int seed, unsigned int n, curandState_t *states);

template <typename T>
void UniformCurand(curandState_t *states, unsigned int n, T *numbers, T a, T b);

template <typename T>
void OneHotCuda(const int batch, const int classes, const T *label, T *onehot);

} // namespace math
} // namespace mlfe
#endif
