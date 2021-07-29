#pragma once
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

namespace dataset{

void read_cifar10_dataset(std::string folder_path,
    std::vector<uint8_t> &train_x,
    std::vector<uint8_t> &train_y,
    std::vector<uint8_t> &valid_x,
    std::vector<uint8_t> &valid_y);

struct cifar10_gen
{
    cifar10_gen(std::vector<uint8_t> images, std::vector<uint8_t> labels)
    {
        __data_size = labels.size();
        __images = images;
        __labels = labels;
    }

    size_t size() const
    {
        return __data_size;
    }

    std::tuple<std::vector<float>, std::vector<float>> operator()(int idx)
    {
        constexpr int classes = 10;
        constexpr int img_size = 32 * 32 * 3;
        std::vector<float> x(img_size);
        std::vector<float> y(classes);
        const auto iter_beg = __images.data() + idx * img_size;
        auto label = (int)__labels[idx];
        std::fill(y.begin(), y.end(), 0.f);
        // one-hot.
        y[label] = 1.f;
        // std::cout<<label<<", "<<idx<<std::endl;
        // convert to float and divide by 255.
        for(int i = 0; i < img_size; ++i)
        {
            x[i] = (float)iter_beg[i] / 255.f;
        }
        x = cvt_channel_last(x);
        return std::make_tuple(x, y);
    }

    std::vector<float> cvt_channel_last(std::vector<float> & x){
        std::vector<float> chann_last(x.size());
        const int H = 32;
        const int W = 32;
        const int C = 3;
        for(int i = 0; i < W; ++i){
            for(int j = 0; j < H; ++j){
                for(int k = 0; k < C; ++k){
                    float val = x[k * H * W + j * W + i];
                    chann_last[j * W * C + i * C + k] = val;
                }
            }
        }
        return chann_last;
    }

    size_t __data_size;
    std::vector<uint8_t> __images;
    std::vector<uint8_t> __labels;
    bool __y_is_x;
};

} // end namespace dataset
