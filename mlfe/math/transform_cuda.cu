#include <cuda.h>
#include <cuda_runtime.h>
#include <cub\block\block_reduce.cuh>
#include "transform.hpp"
#include "../device_context/cuda_context.hpp"

namespace mlfe { namespace math {

template <typename T>
__global__ void maxpool_kernel(
    const int nthreads,
    const T* const x, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    T *y, int *mask) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / pooled_width / pooled_height) % channels;
        const int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        T maxval = -FLT_MAX;
        int maxidx = -1;
        const T* const bottom_slice =
            x + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom_slice[h * width + w] > maxval) {
                    maxidx = h * width + w;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        y[index] = maxval;
        mask[index] = maxidx;
    }
}

template <>
void MaxPool<float, CUDAContext>(
    const int size,
    const float *x, const int c,
    const int h, const int w, const int ph, const int pw, 
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w,
    float *y, int *mask)
{
    maxpool_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS
        >>>(
            size, x, 
            c, h, w, 
            ph, pw, 
            kernel_h, kernel_w, 
            stride_h, stride_w, 
            pad_h, pad_w, 
            y, mask);
}

template <typename T>
__global__ void maxpool_gradient_kernel(
    const int nthreads, 
    const T *dy, const int *mask,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, T *dx)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // find out the local index
        // find out the local offset
        const int w = index % width;
        const int h = (index / width) % height;
        const int c = (index / width / height) % channels;
        const int n = index / width / height / channels;
        const int phstart =
            (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
        const int pwstart =
            (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
        T gradient = 0;
        const int offset = (n * channels + c) * pooled_height * pooled_width;
        const T *top_diff_slice = dy + offset;
        const int *mask_slice = mask + offset;
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                if (mask_slice[ph * pooled_width + pw] == h * width + w) {
                    gradient += top_diff_slice[ph * pooled_width + pw];
                }
            }
        }
        dx[index] = gradient;
    }
}

template <>
void MaxPoolGradient<float, CUDAContext>(
    const int size,
    const float *dy, int *mask,
    const int c, const int h, const int w, 
    const int ph, const int pw, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float *dx)
{
    maxpool_gradient_kernel<float><<<
        CUDA_CONTEXT_GET_BLOCKS(size),
        CUDA_CONTEXT_NUM_THREADS
        >>>(
            size, dy, mask, 
            c, h, w, 
            ph, pw, 
            kernel_h, kernel_w, 
            stride_h, stride_w, 
            pad_h, pad_w, 
            dx
            );
}
} // end namespace math
} // end namespace mlfe
