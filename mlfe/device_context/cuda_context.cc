#include <string>
#include <new>
#include <functional>
#include <cuda_runtime.h>
#include "cuda_context.h"

namespace mlfe {

cublasHandle_t CUDAContext::handler = nullptr;
int CUDAContext::static_shared_counter = 0;
curandGenerator_t CUDAContext::rng = nullptr;

CUDAContext::CUDAContext(){
  if (handler == nullptr) {
    if (cublasCreate(&handler) != cudaSuccess) {
      throw std::string("CUDAContext::CUDAContext() : can not create handler.");
    }
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937);
    curandSetPseudoRandomGeneratorSeed(rng, 1357);
  }
  ++static_shared_counter;
}
CUDAContext::~CUDAContext() {
    --static_shared_counter;

    if (static_shared_counter == 0) {
        if (cublasDestroy(handler) != cudaSuccess) {
            throw std::string("CUDAContext::~CUDAContext() : cuda free handler failed.");
        }
        curandDestroyGenerator(rng);
    }
}

cublasHandle_t CUDAContext::GetHandler() const{
  return handler;
}

} /* namespace mlfe */
