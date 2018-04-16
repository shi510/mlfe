#ifndef __CUDA_CONTEXT_HPP__
#define __CUDA_CONTEXT_HPP__
#include <algorithm>
#include <functional>
#include <cublas_v2.h>
#include "context.hpp"

constexpr int CUDA_CONTEXT_NUM_THREADS = 512;
constexpr int CUDA_CONTEXT_MAXIMUM_NUM_BLOCKS = 2048;

#define CUDA_1D_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;      \
       i < (n);                                            \
       i += blockDim.x * gridDim.x)

inline int CUDA_CONTEXT_GET_BLOCKS(const int N) {
	return std::min<int>((N + CUDA_CONTEXT_NUM_THREADS - 1) / CUDA_CONTEXT_NUM_THREADS,
		CUDA_CONTEXT_MAXIMUM_NUM_BLOCKS);
}

namespace mlfe {

class CUDAContext final : public Context {
public:
	CUDAContext();

	~CUDAContext() override;

	void * GetDevicePtr() const override;

	cublasHandle_t GetHandler() const;

	void Clear() override;

	int Size() const override;

protected:
	void Allocator(
                  const unsigned int size,
                  const unsigned int block_size
                  ) override;

	void CopyTo(
              const unsigned int offset,
              const unsigned int size,
              const unsigned int block_size,
              void *to
              ) const override;

	void CopyFrom(
                const unsigned int offset,
                const unsigned int size,
                const unsigned int block_size,
                const void *from
                ) override;

private:
	void *ptr_;
	int size_;
	static int static_shared_counter;
	static cublasHandle_t handler;
};/* class CUDAContext */

} /* namespace mlfe */
#endif /*__CUDA_CONTEXT_HPP__*/
