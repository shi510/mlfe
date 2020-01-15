#include "activations.h"
#include "mlfe/device_context/cuda_context.h"
#include <third_party/cub/cub/block/block_reduce.cuh>

namespace mlfe{ namespace math{

template <class DataType>
__global__ void relu_kernel(const int size,
                            const DataType *x,
                            DataType *y
                           )
{
    CUDA_1D_KERNEL_LOOP(i, size){
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

template <class DataType>
__global__ void relu_gradient_kernel(const int size,
                                     const DataType *x,
                                     const DataType *dy,
                                     DataType *dx
                                    )
{
    CUDA_1D_KERNEL_LOOP(i, size){
        dx[i] = x[i] > 0 ? dy[i] : 0;
    }
}

template <>
void relu<float, CUDAContext>(const int size,
                              const float *x,
                              float *y
                             )
{
    relu_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS >> >(size, x, y);
}

template <>
void relu_gradient<float, CUDAContext>(const int size,
                                       const float *x,
                                       const float *dy,
                                       float *dx
                                      )
{
    relu_gradient_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x, dy, dx);
}


template <class DataType>
__global__ void sigmoid_kernel(const int size,
                               const DataType *x,
                               DataType *y
                              )
{
    CUDA_1D_KERNEL_LOOP(i, size){
        y[i] = 1.f /(1.f + exp(-x[i]));
    }
}

template <class DataType>
__global__  void sigmoid_gradient_kernel(const int size,
                                         const DataType *y,
                                         const DataType *dy,
                                         DataType *dx
                                        )
{
    CUDA_1D_KERNEL_LOOP(i, size){
        dx[i] = dy[i] * y[i] *(1. - y[i]);
    }
}

template <>
void sigmoid<float, CUDAContext>(const int size,
                                 const float *x,
                                 float *y
                                )
{
    sigmoid_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, x, y);
}

template <>
void sigmoid_gradient<float, CUDAContext>(const int size,
                                          const float *y,
                                          const float *dy,
                                          float *dx
                                         )
{
    sigmoid_gradient_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, y, dy, dx);
}

template <class DataType> __global__
void xent_kernel(const int N,
                 const int D,
                 const DataType *Pdata,
                 const DataType *labeldata,
                 DataType* Ydata
                )
{
    typedef cub::BlockReduce<float, CUDA_CONTEXT_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for(int i = blockIdx.x; i < N; i += gridDim.x){
        DataType sum = DataType(0);
        for(int j = threadIdx.x; j < D; j += blockDim.x){
            int idx = i * D + j;
            sum += log(max(Pdata[idx], DataType(1e-20))) * labeldata[idx];
        }

        DataType tot = BlockReduce(temp_storage).Sum(sum);
        __syncthreads();
        if(threadIdx.x == 0){
            Ydata[i] = -tot;
        }
        __syncthreads();
    }
}

template <>
void cross_entropy<float, CUDAContext>(const int m,
                                       const int n,
                                       const float *prob_ptr,
                                       const float *label_ptr,
                                       float *loss_ptr
                                      )
{
    xent_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr);
}

template <> void
cross_entropy<double, CUDAContext>(const int m,
                                   const int n,
                                   const double *prob_ptr,
                                   const double *label_ptr,
                                   double *loss_ptr
                                  )
{
    xent_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr);
}

template <class DataType> __global__
void xent_gradient_kernel(const int N,
                          const int D,
                          const DataType* Pdata,
                          const DataType* labeldata,
                          const DataType* lossdata,
                          DataType* dXdata
                         )
{
    CUDA_1D_KERNEL_LOOP(n, N * D){
        int idx = n / D;
        dXdata[n] = (Pdata[n] - labeldata[n]) * lossdata[idx];
    }
}

template <> void
cross_entropy_gradient<float, CUDAContext>(const int m,
                                           const int n,
                                           const float *prob_ptr,
                                           const float *label_ptr,
                                           const float *loss_ptr,
                                           float *dx_ptr
                                          )
{
    xent_gradient_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr, dx_ptr);
}

template <> void
cross_entropy_gradient<double, CUDAContext>(const int m, const int n,
                                            const double *prob_ptr,
                                            const double *label_ptr,
                                            const double *loss_ptr,
                                            double *dx_ptr
                                           )
{
    xent_gradient_kernel<double><<<CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, prob_ptr, label_ptr, loss_ptr, dx_ptr);
}

template <class DataType> __device__
float sigmoid_xent_forward_kernel(const DataType x,
                                  const DataType t
                                 )
{
    return x * t - max(x, DataType(0)) - log(DataType(1) + std::exp(-abs(x)));
}


template <class DataType> __global__
void sigmoid_xent_kernel(const int m,
                         const int n,
                         const DataType *x_ptr,
                         const DataType *t_ptr,
                         DataType *loss_ptr
                        )
{
    int i = blockIdx.x;
    int last_idx =(i + 1) * n;
    DataType value = 0;
    for(int in_idx = i * n + threadIdx.x; in_idx < last_idx; in_idx += blockDim.x){
        value += sigmoid_xent_forward_kernel(x_ptr[in_idx], t_ptr[in_idx]);
    }

    typedef cub::BlockReduce<DataType, CUDA_CONTEXT_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    DataType sum = BlockReduce(temp_storage).Sum(value);
    if(threadIdx.x == 0){
        loss_ptr[i] = -sum / static_cast<DataType>(n);
    }
}

template <> void
sigmoid_cross_entropy<float, CUDAContext>(const int m, const int n,
                                          const float *x_ptr,
                                          const float *t_ptr,
                                          float *loss_ptr
                                         )
{
    sigmoid_xent_kernel<float><<<m,
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, x_ptr, t_ptr, loss_ptr);
}

template <class DataType> __global__
void sigmoid_xent_gradient_kernel(const int m,
                                  const int n,
                                  const DataType *x_ptr,
                                  const DataType *t_ptr,
                                  const DataType *dy_ptr,
                                  DataType *dx_ptr
                                 )
{
    CUDA_1D_KERNEL_LOOP(index, m * n){
        int t = index / n;
        DataType dy = dy_ptr[t] / DataType(n);
        DataType sig = DataType(1) /(DataType(1) + std::exp(-x_ptr[index]));
        dx_ptr[index] =(sig - t_ptr[index]) * dy;
    }
}

template <> void
sigmoid_cross_entropy_gradient<float, CUDAContext>(const int m,
                                                   const int n,
                                                   const float *x_ptr,
                                                   const float *t_ptr,
                                                   const float *dy_ptr,
                                                   float *dx_ptr
                                                  )
{
    sigmoid_xent_gradient_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(m * n),
        CUDA_CONTEXT_NUM_THREADS>>>(m, n, x_ptr, t_ptr, dy_ptr, dx_ptr);
}

template <class DataType> __global__
void reduce_mean_gradient_kernel(const int size,
                                 const DataType scale,
                                 const DataType *dy,
                                 DataType *dx
                                )
{
    DataType dy_val = dy[0];
    CUDA_1D_KERNEL_LOOP(index, size){
        dx[index] = dy_val * scale;
    }
}

template <>
void reduce_mean_gradient<float, CUDAContext>(const int size,
                                              const float scale,
                                              const float *dy_ptr,
                                              float *dx_ptr
                                             )
{
    reduce_mean_gradient_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS>>>(size, scale, dy_ptr, dx_ptr);
}

} // end namespace math
} // end namespace mlfe
