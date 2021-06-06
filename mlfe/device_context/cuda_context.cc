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
        handler = nullptr;
        curandDestroyGenerator(rng);
    }
}

cublasHandle_t CUDAContext::GetHandler() const{
    return handler;
}

cublasHandle_t cuda_context_v2::cublas_handle = nullptr;
cudnnHandle_t cuda_context_v2::cudnn_handle = nullptr;
int cuda_context_v2::static_shared_counter = 0;

cuda_context_v2::cuda_context_v2(){
    if (cublas_handle == nullptr) {
        if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::string("cuda_context_v2::cuda_context_v2() : can not create cublas handle.");
        }
    }
    if (cudnn_handle == nullptr) {
        if (cudnnCreate(&cudnn_handle) != CUDNN_STATUS_SUCCESS) {
            throw std::string("cuda_context_v2::cuda_context_v2() : can not create cudnn handle.");
        }
    }
    ++static_shared_counter;
}

cuda_context_v2::~cuda_context_v2() {
    --static_shared_counter;

    if (static_shared_counter == 0) {
        if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::string("cuda_context_v2::~cuda_context_v2() : cuda free cublas handle failed.");
        }
        if (cudnnDestroy(cudnn_handle) != CUDNN_STATUS_SUCCESS) {
            throw std::string("cuda_context_v2::~cuda_context_v2() : cuda free cudnn handle failed.");
        }
        cublas_handle = nullptr;
        cudnn_handle = nullptr;
    }
}

cublasHandle_t cuda_context_v2::get_cublas_handle() const {
    return cublas_handle;
}

cudnnHandle_t cuda_context_v2::get_cudnn_handle() const {
    return cudnn_handle;
}

std::shared_ptr<cuda_context_v2> cuda_context_v2::create()
{
    static std::shared_ptr<cuda_context_v2> cuda_ctx(new cuda_context_v2());

    return cuda_ctx;
}

} /* namespace mlfe */
