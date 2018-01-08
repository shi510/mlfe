#include <string>
#include <new>
#include <functional>
#include <cuda_runtime.h>
#include "cuda_context.hpp"

namespace mlfe {

cublasHandle_t CUDAContext::handler = nullptr;
int CUDAContext::static_shared_counter = 0;

CUDAContext::CUDAContext() {
  if (handler == nullptr) {
    if (cublasCreate(&handler) != cudaSuccess) {
      throw std::string("CUDAContext::CUDAContext() : can not create handler.");
    }
  }
  ++static_shared_counter;
}
CUDAContext::~CUDAContext() {
  Clear();
}

void * CUDAContext::GetDevicePtr() const {
  return ptr_;
}

cublasHandle_t CUDAContext::GetHandler() {
  return handler;
}

void CUDAContext::Clear() {
  --static_shared_counter;
  if (static_shared_counter == 0) {
    if (cudaFree(ptr_) != cudaSuccess) {
      throw std::string("CUDAContext::Clear() : cuda free memory failed.");
    }
    ptr_ = nullptr;
  }
}

int CUDAContext::Size() const {
  return size_;
}

void CUDAContext::Allocator(
                            const unsigned int size,
                            const unsigned int block_size
                            ){
  size_ = size * block_size;
  if (cudaMalloc((void**)&ptr_, size_) != cudaSuccess) {
  throw std::string("CUDAContext::Allocator() : device memory allocation failed.");
  }
}

void CUDAContext::CopyTo(
                          const unsigned int offset,
                          const unsigned int size,
                          const unsigned int block_size,
                          void *to
                          ){
  if (size * block_size > size_) {
  throw std::string("Copy size is bigger than allocated device memory.");
  }
  if (cudaMemcpy(to, ptr_, size * block_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
  throw std::string("CUDAContext::CopyTo() : device to host copy failed.");
  }
}

void CUDAContext::CopyFrom(
                            const unsigned int offset,
                            const unsigned int size,
                            const unsigned int block_size,
                            const void *from
  ){
  if (size * block_size > size_) {
  throw std::string("Copy size is bigger than allocated device memory.");
  }
  if (cudaMemcpy(ptr_, from, size * block_size, cudaMemcpyHostToDevice) != cudaSuccess) {
  throw std::string("CUDAContext::CopyFrom() : host to device copy failed.");
  }
}

} /* namespace mlfe */
