#ifndef __TRANSFORM_HPP__
#define __TRANSFORM_HPP__

namespace mlfe{ namespace math{

template <class DataType, class DeviceContext>
void im2col(const int im_c, const int im_h, const int im_w,
            const int kernel_h, const int kernel_w,
            const int stride, const int padding,
            const DataType *_im, DataType *_col
            );

template <class DataType, class DeviceContext>
void col2im(DataType* data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad,
            DataType* data_im
            );

} /* namespace math */
} /* namespace mlfe */
#endif /* __TRANSFORM_HPP__ */
