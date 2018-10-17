#include <cuda.h>
#include <cuda_runtime.h>
#include <cub\block\block_reduce.cuh>
#include "transform.h"
#include "../device_context/cuda_context.h"

namespace mlfe { namespace math {
template <class DataType>
__global__ void im2col_kernel(const int n, const DataType* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int pad_t, const int pad_l,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    DataType* data_col) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * kernel_h * kernel_w;
        int h_in = h_out * stride_h - pad_t;
        int w_in = w_out * stride_w - pad_l;
        DataType* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const DataType* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_in + i * dilation_h;
                int w = w_in + j * dilation_w;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

template <>
void im2col<float, CUDAContext>(const int channel,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const float *im_ptr,
    float *col_ptr
    ) {
    const int out_height = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_w) / stride + 1;
    const int num_kernel = channel * out_height * out_width;

    im2col_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(num_kernel),
        CUDA_CONTEXT_NUM_THREADS>>>(
            num_kernel, im_ptr, 
            height, width, 
            kernel_h, kernel_w, 
            1, 1, 
            padding, padding, 
            stride, stride, 
            out_height, out_width, 
            col_ptr
            );
}

template <typename DataType>
__global__ void col2im_kernel(const int n, const DataType* data_col,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int dilation_h, const int dilation_w,
    const int pad_t, const int pad_l,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    DataType* data_im) {

    const int dpatch_h = dilation_h * (patch_h - 1) + 1;
    const int dpatch_w = dilation_w * (patch_w - 1) + 1;

    CUDA_1D_KERNEL_LOOP(index, n) {
        DataType val = 0;
        int w = index % width + pad_l;
        int h = (index / width) % height + pad_t;
        int c = index / (width * height);

        int w_col_start = (w < dpatch_w) ? 0 : (w - dpatch_w) / stride_w + 1;
        int w_col_end = min(w / stride_w + 1, width_col);
        int h_col_start = (h < dpatch_h) ? 0 : (h - dpatch_h) / stride_h + 1;
        int h_col_end = min(h / stride_h + 1, height_col);

        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                int h_k = (h - h_col * stride_h);
                int w_k = (w - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index =
                        (((c * patch_h + h_k) * patch_w + w_k) * height_col + h_col) *
                        width_col +
                        w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}

template <>
void col2im<float, CUDAContext>(float* data_col,
    int channels,
    int height,
    int width,
    int ksize,
    int stride,
    int pad,
    float* data_im
    ) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int num_kernel = channels * height * width;
    col2im_kernel<float><<<CUDA_CONTEXT_GET_BLOCKS(num_kernel),
        CUDA_CONTEXT_NUM_THREADS>>>(
            num_kernel, data_col, 
            height, width, 
            ksize, ksize, 
            1, 1, 
            pad, pad, 
            stride, stride, 
            height_col, width_col, 
            data_im
            );
}

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
