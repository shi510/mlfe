#include "mnist_models.h"
#include <mlfe/core.h>
#include <mlfe/operators.h>
#include <mlfe/optimizers.h>
#include <mlfe/utils/db/simpledb_reader.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace train_example{
using namespace mlfe;
namespace fn = functional;

AutoEncoder::AutoEncoder(const int batch, 
                         const double lr, 
                         const double mm) 
    : sgd(fn::create_gradient_descent_optimizer(lr, mm)){
    build(batch);
    rng = std::mt19937(std::random_device()());
}

std::vector<float> AutoEncoder::recon(const std::vector<float> &x_val){
    using Floats = std::vector<float>;
    const float *recon_ptr;
    std::copy(x_val.begin(), x_val.end(), x.begin<float>());
    decode_with_sigmoid.eval();
    recon_ptr = decode_with_sigmoid.data<float>();
    return Floats(recon_ptr, 
                  recon_ptr + decode_with_sigmoid.size());
}

void AutoEncoder::forward(const std::vector<float> &x_val){
    std::copy(x_val.begin(), x_val.end(), x.begin<float>());
    loss.eval();
}

void AutoEncoder::backward(){
    loss.backprop();
}

std::vector<float> AutoEncoder::get_loss(const std::vector<float> &x_val){
    using Floats = std::vector<float>;
    const float *loss_ptr;
    std::copy(x_val.begin(), x_val.end(), x.begin<float>());
    loss.eval();
    loss_ptr = loss.data<float>();
    return Floats(loss_ptr, loss_ptr + loss.size());
}

void AutoEncoder::build(const int batch){
    x = fn::create_variable({64, 28 * 28});
    encode = encoder(x);
    decode = decoder(encode);
    decode_with_sigmoid = fn::sigmoid(decode);
    loss = fn::mean(fn::sigmoid_cross_entropy(decode, x));
}

Tensor AutoEncoder::encoder(Tensor x){
    Tensor en = fc_relu("en0", x, 500, 1e-2, true);
    en = fc_relu("en1", en, 300, 1e-2, true);
    en = fc_relu("en2", en, 150, 1e-1, true);
    en = fc_relu("en3", en, 50, 1e-1, true);
    en = fc_relu("en4", en, 25, 1e-1, false);
    return en;
}

Tensor AutoEncoder::decoder(Tensor x){
    Tensor de = fc_relu("de0", x, 50, 1e-1, true);
    de = fc_relu("de1", de, 150, 1e-1, true);
    de = fc_relu("de2", de, 300, 1e-1, true);
    de = fc_relu("de3", de, 500, 1e-2, true);
    de = fc_relu("de4", de, 784, 1e-2, false);
    return de;
}

Tensor AutoEncoder::fc_relu(const std::string name, 
                            Tensor x, 
                            const int out, 
                            const double std,
                            const bool is_act
                           ){
    auto random_fn = [this, &std](){
        auto dist = std::normal_distribution<float>(0, std);
        return dist(rng);
    };
    int in = x.size() / x.shape()[0];
    Tensor w = fn::create_variable({in, out});
    Tensor b = fn::create_variable({out});
    Tensor fc = fn::add(fn::matmul(x, w), b);
    if(is_act){
        fc = fn::relu(fc);
    }
    vars[name + "_w"] = w;
    vars[name + "_b"] = b;
    std::fill(b.mutable_data<float>(), 
              b.mutable_data<float>() + b.size(),
              0.1);
    std::generate(w.mutable_data<float>(), 
                  w.mutable_data<float>() + w.size(),
                  random_fn);
    return fc;
}

void AutoEncoder::update(){
    for(auto &it : vars){
        sgd->apply(it.second, it.second.grad());
    }
}

Lenet::Lenet(const int batch, const double lr, const double mm) 
    : sgd(fn::create_gradient_descent_optimizer(lr, mm)){
    build(batch);
}

void Lenet::forward(const std::vector<float> &x_val,
                    const std::vector<float> &y_val){
    std::copy(x_val.begin(), x_val.end(), x.begin<float>());
    std::copy(y_val.begin(), y_val.end(), y.begin<float>());
    loss.eval();
}

void Lenet::backward(){
    loss.backprop();
}

Lenet::LogitLoss Lenet::get_logit_loss(const std::vector<float> &x_val,
                                       const std::vector<float> &y_val){
    using Floats = std::vector<float>;
    Floats logit_val, loss_val;
    const float *logit_ptr, *loss_ptr;
    std::copy(x_val.begin(), x_val.end(), x.begin<float>());
    std::copy(y_val.begin(), y_val.end(), y.begin<float>());
    loss.eval();
    logit_ptr = logit.data<float>();
    loss_ptr = loss.data<float>();
    logit_val.assign(logit_ptr, logit_ptr + logit.size());
    loss_val.assign(loss_ptr, loss_ptr + loss.size());
    return std::make_tuple(logit_val, loss_val);
}

void Lenet::build(const int batch){
    x = fn::create_variable({batch, 1, 28, 28});
    y = fn::create_variable({batch, 10});
    logit = fn::relu(conv2d("conv1", x, 16, 5, 1, 0, 1e-1));
    logit = maxpool("maxpool1", logit, 2, 2, 0);
    logit = fn::relu(conv2d("conv2", logit, 32, 5, 1, 0, 1e-1));
    logit = maxpool("maxpool2", logit, 2, 2, 0);
    logit = fn::reshape(logit, {batch, 4 * 4 * 32});
    logit = fc_relu("fc1", logit, 128, 1e-1, true);
    logit = fc_relu("fc2", logit, 10, 1e-2, false);
    loss = fn::mean(fn::softmax_cross_entropy(logit, y));
}

mlfe::Tensor Lenet::fc_relu(const std::string name,
                            const mlfe::Tensor x,
                            const int out,
                            const double std,
                            const bool is_act
                           ){
    auto kaiming_he_fn = [this, &x](){
        float in = (float)x.size() / x.shape()[0];
        float std = std::sqrt(2.f / (in));
        auto dist = std::normal_distribution<float>(0, std);
        return dist(rng);
    };
    int in = x.size() / x.shape()[0];
    Tensor w = fn::create_variable({in, out});
    Tensor b = fn::create_variable({out});
    Tensor fc = fn::add(fn::matmul(x, w), b);
    if(is_act){
        fc = fn::relu(fc);
    }
    vars[name + "_w"] = w;
    vars[name + "_b"] = b;
    std::fill(b.begin<float>(), b.end<float>(), 0.1);
    std::generate(w.begin<float>(), w.end<float>(), kaiming_he_fn);
    return fc;
}

mlfe::Tensor Lenet::conv2d(const std::string name,
                           const mlfe::Tensor x,
                           const int filter,
                           const int kernel,
                           const int stride,
                           const int padding,
                           const double std
                          ){
    auto kaiming_he_fn = [this, &x](){
        float in = (float)x.size() / x.shape()[0];
        float std = std::sqrt(2.f / (in));
        auto dist = std::normal_distribution<float>(0, std);
        return dist(rng);
    };
    Tensor w = fn::create_variable({filter, x.shape()[1], kernel, kernel});
    Tensor b = fn::create_variable({1, filter, 1, 1});
    Tensor y = fn::conv2d(x, w, {stride, stride}, {padding, padding});
    y = fn::add(y, fn::broadcast(b, y.shape()));
    vars[name + "_w"] = w;
    vars[name + "_b"] = b;
    std::generate(w.begin<float>(), w.end<float>(), kaiming_he_fn);
    std::fill(b.begin<float>(), b.end<float>(), 0.1f);
    return y;
}

mlfe::Tensor Lenet::maxpool(const std::string name,
                            const mlfe::Tensor x,
                            const int kernel,
                            const int stride,
                            const int padding
                           ){
    Tensor y = fn::pool_max(x,
                            {kernel, kernel}, 
                            {stride, stride}, 
                            {padding, padding}
                           );
    return y;
}

void Lenet::update(){
    for(auto &it : vars){
        sgd->apply(it.second, it.second.grad());
    }
}

} // end namespace train_example
