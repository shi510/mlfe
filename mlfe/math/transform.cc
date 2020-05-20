#include "transform.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{ namespace math{

template <>
void im2col<float, CPUContext>(const int channel,
                               const int height,
                               const int width,
                               const int kernel_h,
                               const int kernel_w,
                               const int stride,
                               const int padding,
                               const float *im_ptr,
                               float *col_ptr
                               ){
    const int channels_col = channel * kernel_w * kernel_h;
    const int out_height = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_w) / stride + 1;
    
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * out_height + h) * out_width + w;
                im_row -= padding;
                im_col -= padding;
                if(im_row < 0 || im_col < 0 || im_row >= height || im_col >= width){
                    col_ptr[col_index] = 0;
                }
                else{
                    col_ptr[col_index] = im_ptr[im_col + width * (im_row + height * c_im)];
                }
            }
        }
    }
}

template <>
void im2col<double, CPUContext>(const int channel,
                                const int height,
                                const int width,
                                const int kernel_h,
                                const int kernel_w,
                                const int stride,
                                const int padding,
                                const double *im_ptr,
                                double *col_ptr
                                ){
    const int channels_col = channel * kernel_w * kernel_h;
    const int out_height = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_w) / stride + 1;
    
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * out_height + h) * out_width + w;
                im_row -= padding;
                im_col -= padding;
                if(im_row < 0 || im_col < 0 || im_row >= height || im_col >= width){
                    col_ptr[col_index] = 0;
                }
                else{
                    col_ptr[col_index] = im_ptr[im_col + width * (im_row + height * c_im)];
                }
            }
        }
    }
}

template <class DataType>
void col2im_add_pixel(DataType *im, int height, int width, int channels,
                      int row, int col, int channel, int pad, DataType val){
    row -= pad;
    col -= pad;
    
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

template <>
void col2im<float, CPUContext>(float* data_col,
                               int channels,
                               int height,
                               int width,
                               int ksize,
                               int stride,
                               int pad,
                               float* data_im
                               ){
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float val = data_col[col_index];
                col2im_add_pixel<float>(data_im, height, width, channels,
                                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

template <>
void col2im<double, CPUContext>(double* data_col,
                                int channels,
                                int height,
                                int width,
                                int ksize,
                                int stride,
                                int pad,
                                double* data_im
                                ){
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float val = data_col[col_index];
                col2im_add_pixel<double>(data_im, height, width, channels,
                                         im_row, im_col, c_im, pad, val);
            }
        }
    }
}

std::vector<int>
check_broadcasting(std::vector<int>* a, std::vector<int>* b)
{
    std::vector<int> shape;
    int max = std::max(a->size(), b->size());
    while (max != a->size()) {
        a->insert(a->begin(), 1);
    }
    while (max != b->size()) {
        b->insert(b->begin(), 1);
    }
    shape.resize(max);
    for (int n = max - 1; n >= 1; --n) {
        int a_at = a->at(n);
        int b_at = b->at(n);
        if (a_at != 1 && b_at != 1 && a_at != b_at) {
            return std::vector<int>();
        }
        else {
            shape[n] = std::max(a_at, b_at);
        }
    }
    shape[0] = std::max(a->at(0), b->at(0));
    return shape;
}

template <>
void broadcast<float, CPUContext>(const float* x, float* y,
    int Nx, int Cx, int Hx, int Wx,
    int Ny, int Cy, int Hy, int Wy)
{
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Cy; ++j) {
            for (int k = 0; k < Hy; ++k) {
                for (int l = 0; l < Wy; ++l) {
                    int x_idx = (i % Nx) * Cx * Hx * Wx +
                        (j % Cx) * Hx * Wx +
                        (k % Hx) * Wx +
                        (l % Wx);
                    int y_idx = i * Cy * Hy * Wy +
                        j * Hy * Wy +
                        k * Wy +
                        l;
                    y[y_idx] = x[x_idx];
                }
            }
        }
    }
}

template <>
void broadcast_gradient<float, CPUContext>(const float* dy, float* dx,
    int Ny, int Cy, int Hy, int Wy,
    int Nx, int Cx, int Hx, int Wx)
{
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Cy; ++j) {
            for (int k = 0; k < Hy; ++k) {
                for (int l = 0; l < Wy; ++l) {
                    int dx_idx = (i % Nx) * Cx * Hx * Wx +
                        (j % Cx) * Hx * Wx +
                        (k % Hx) * Wx +
                        (l % Wx);
                    int dy_idx = i * Cy * Hy * Wy +
                        j * Hy * Wy +
                        k * Wy +
                        l;
                    dx[dx_idx] += dy[dy_idx];
                }
            }
        }
    }
}

} /* math */
} /* mlfe */
