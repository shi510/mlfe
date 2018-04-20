#include "functions_cuda.hpp"

namespace mlfe { namespace math {

__global__ void curand_init_kernel(unsigned int seed, curandState_t *states) {
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

template <typename T>
__global__ void curand_uniform_kernel(curandState_t *states, T *numbers, T a, T b) {
    numbers[blockIdx.x] = curand_uniform(&states[blockIdx.x]);
    numbers[blockIdx.x] = numbers[blockIdx.x] * (b - a) + a;
}

void InitCurand(unsigned int seed, unsigned int n, curandState_t* states) {
    curand_init_kernel<<<n, 1>>>(seed, states);
}

template <>
void UniformCurand<float>(curandState_t *states, unsigned int n, float *numbers, float a, float b) {
    curand_uniform_kernel<float><<<n, 1>>>(states, numbers, a, b);
}

template <>
void UniformCurand<double>(curandState_t *states, unsigned int n, double *numbers, double a, double b) {
    curand_uniform_kernel<double><<<n, 1>>>(states, numbers, a, b);
}

} // end namespace math
} // end namespace mlfe
