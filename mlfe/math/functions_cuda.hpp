#ifndef __MATH_FUNCTIONS_CUDA_HPP__
#define __MATH_FUNCTIONS_CUDA_HPP__
#include <curand_kernel.h>

namespace mlfe { namespace math {

void InitCurand(unsigned int seed, unsigned int n, curandState_t *states);

template <typename T>
void UniformCurand(curandGenerator_t *gen, const unsigned int size, T *numbers, const T a, const T b);

template <typename T>
void OneHotCuda(const int batch, const int classes, const T *label, T *onehot);

template <typename T>
void AccuracyCuda(const int batch, const int classes, const int top_k, const T *prob, const T *label, T *accuracy);

} // namespace math
} // namespace mlfe
#endif
