
# mlfe : Modeling Learnable Feature Extractor  
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.com/shi510/mlfe.svg?branch=master)](https://travis-ci.com/shi510/mlfe)  

## Why do I develop this repo?
Just curious.  

## Roadmap for the future.
(1) Neural Architecture Search  
 - New search space including low-bit quantization and sparsity.  

## Index
1. [Basic Example](#Basic-Example)
2. [Simple neural network for MNIST dataset](#Simple-neural-network-for-MNIST-dataset)
3. [Auto Encoder](#Auto-Encoder)
4. [Convolutional neural network](#Convolutional-neural-network)
5. [Deep CNN for CIFAR10](#Deep-CNN-for-CIFAR10)
6. [The easiest way to build a model](#The-easiest-way-to-build-a-model)
7. [How to compile](#How-to-compile)
8. [Supported operators](#Supported-operators)

## Basic Example
You can create a variable using create_variable function.  
The first parameter of the function is shape.  
Also you can create a variable filled with constant value using constant function.  
You can modify the variable's contents by using standard c++ function std::fill or can acceess variable's data address pointer.  
If you access the address pointer using mutable_data, the variable synchronizes with the device memory.  
The device means GPU device, but if you compiled with CPU, it does not synchronized.  

```c++
using namespace mlfe;
namespace fn = functional;

auto x = fn::create_variable({2, 2});
auto two = Tensor::from_vector<float>({2, 2, 2, 2}, {2, 2});
std::fill(x.begin<float>(), x.end<float>(), 1);
// same result with std::fill, accessing address pointer directly.
for(int n = 0; n < x.size(); ++n){
    x.mutable_data<float>()[n] = 1;
}
```
Here, we build simple operations that is 3 * (x + 2)^2 and then we apply mean function.  
```c++
auto y = one + two;
y = 3 * (y * y);
y = fn::reduce_mean(y); // result is 27
```

A gradient of variable can calculate by calling `backprop` function.  
You can access the gradient by calling `grad` function.  
The gradient of one is 4.5.  
```c++
result.backprop();
x.grad(); // its gradients are {{4.5, 4.5}, {4.5, 4.5}}
```

## Simple neural network for MNIST dataset

To train mnist data, we build a simple neural network.  
This code is in [example/mnist_v2](example/mnist_v2).  

*mnist data -> fully connected NN -> softmax.*  

First step is to including headers.  
```c++
#include <mlfe/core/tensor.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/operators/softmax_cross_entropy.h>
#include <mlfe/operators/matmul.h>
```
For convenience, we use namespace abbreviation.  
```c++
using namespace mlfe;
namespace fn = functional;
namespace op = operators;
```

1. Define forward function.  
2. Define criterion function for training loss.  
3. Define weights update function.  

```c++
struct mnist_simple_net
{
    Tensor weights;
    Tensor biases;

    mnist_simple_net(int input_size, int out_size)
    {
        weights = fn::create_variable({input_size, out_size});
        biases = fn::create_variable({out_size});
        std::fill(weights.begin<float>(), weights.end<float>(), 0.f);
        std::fill(biases.begin<float>(), biases.end<float>(), 0.1f);
    }

    Tensor forward(Tensor x)
    {
        auto logits = op::matmul(x, weights);
        return logits + biases;
    }

    Tensor criterion(Tensor y_true, Tensor y_pred)
    {
        auto loss = op::softmax_cross_entropy(y_true, y_pred);
        loss = op::reduce_mean(loss);
        return loss;
    }

    void update_weights(float lr)
    {
        weights -= lr * weights.grad_v2();
        biases -= 2.f * lr * biases.grad_v2();
    }

    void zero_grad()
    {
        weights.grad_v2().zero();
        biases.grad_v2().zero();
    }
};
```

1. Define your training loop.  
2. Call the functinos :  
    (zero_grad -> forward -> criterion -> backprop -> update_weights)  

Gradients are accumulated if you not make the gradient zero.  

```c++
void train_simplenet(
    dataset::mnist_gen train_set,
    dataset::mnist_gen valid_set)
{
    const int BATCH = 32;
    const int EPOCH = 1;
    const int INPUT_SIZE = 28 * 28;
    const int OUTPUT_SIZE = 10;
    const int NUM_ITER = train_set.size() / BATCH;

    auto model = models::simple_net(INPUT_SIZE, OUTPUT_SIZE);
    auto images = std::vector<float>(BATCH*INPUT_SIZE);
    auto labels = std::vector<float>(BATCH*OUTPUT_SIZE);
    for(int e = 0; e < EPOCH; ++e){
        float mean = 0.f;
        for(int i = 0; i < NUM_ITER; ++i){
            fill_batch(train_set, images, labels, BATCH, i);
            model.zero_grad();
            auto x = Tensor::from_vector(images, {BATCH, INPUT_SIZE});
            auto y_true = Tensor::from_vector(labels, {BATCH, OUTPUT_SIZE});
            auto y_pred = model.forward(x);
            auto loss = model.criterion(y_true, y_pred);
            loss.backprop_v2();
            model.update_weights(1e-1);
            mean += loss.data<float>()[0];
        }
        std::cout<<"EPOCH "<<e + 1<<" : "<<mean / NUM_ITER<<std::endl;
    }
}
```
After 1000 iterations are finished, the variables will close to the optimal solution.  
This model performs about 90% acccuracy on MNIST 10K test images.  
The following figure shows visualization of the w variable, the Red colour represents negative value, the Blue colour represents positive value.  

![visualization weights](https://raw.githubusercontent.com/shi510/mlfe/master/figures/fig_mnist_weights.jpg)

## Auto Encoder
```c++
#include <mlfe/core/tensor.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/operators/squared_difference.h>
#include <mlfe/operators/sigmoid.h>
#include <mlfe/nn/layers/linear.h>
#include <mlfe/nn/module.h>
using namespace mlfe;
namespace op = mlfe::operators;

struct encoder : nn::module{
    nn::linear fc1;
    nn::linear fc2;
    nn::linear fc3;
    nn::linear fc4;

    encoder(){
        fc1 = trainable(nn::linear(28*28, 300));
        fc2 = trainable(nn::linear(300, 150));
        fc3 = trainable(nn::linear(150, 50));
        fc4 = trainable(nn::linear(50, 10));
    }

    Tensor forward(Tensor x){
        x = fc1(x);
        x = op::sigmoid(x);
        x = fc2(x);
        x = op::sigmoid(x);
        x = fc3(x);
        x = op::sigmoid(x);
        x = fc4(x);
        x = op::sigmoid(x);
        return x;
    }
};

struct decoder : nn::module{
    nn::linear fc1;
    nn::linear fc2;
    nn::linear fc3;
    nn::linear fc4;

    decoder(){
        fc1 = trainable(nn::linear(10, 50));
        fc2 = trainable(nn::linear(50, 150));
        fc3 = trainable(nn::linear(150, 300));
        fc4 = trainable(nn::linear(300, 28*28));
    }

    Tensor forward(Tensor x){
        x = fc1(x);
        x = op::sigmoid(x);
        x = fc2(x);
        x = op::sigmoid(x);
        x = fc3(x);
        x = op::sigmoid(x);
        x = fc4(x);
        x = op::sigmoid(x);
        return x;
    }
};

struct autoencoder : nn::module{
    encoder encode;
    decoder decode;

    autoencoder()
    {
        encode = trainable(encoder());
        decode = trainable(decoder());
    }

    Tensor forward(Tensor x)
    {
        x = encode.forward(x);
        x = decode.forward(x);
        return x;
    }

    Tensor criterion(Tensor y_true, Tensor y_pred)
    {
        auto loss = op::squared_difference(y_true, y_pred);
        return op::reduce_mean(loss);
    }
};
```

## Convolutional neural network

1. Inherit the nn::module.  
2. Create layers in your class.  
3. Notify trainable variables to nn::module by enclosing a layer with `trainable` function.  

The `trainable` function finds trainables in a layer and collects it.  
See [mnist example](example/mnist_v2).  
**The accuracy of this model on mnist dataset is about 98% at 2 epoch.**  

```c++
using namespace mlfe;
namespace op = mlfe::operators;

struct mnist_conv_net : nn::module{
    nn::conv2d conv1;
    nn::conv2d conv2;
    nn::linear fc1;
    nn::linear fc2;

    mnist_conv_net(){
        conv1 = trainable(nn::conv2d(/*input channel=*/1,
                                     /*output channel=*/16,
                                     /*kernel size=*/{3, 3},
                                     /*stride size=*/{1, 1},
                                     /*output size same with inputs=*/true));
        conv2 = trainable(nn::conv2d(16, 32, {3, 3}, {1, 1}, true));
        fc1 = trainable(nn::linear(
                                   /*input channel=*/7*7*32,
                                   /*output channel=*/256));
        fc2 = trainable(nn::linear(256, 10));
    }

    tensor forward(tensor x, bool is_training=false){
        x = conv1(x);
        x = op::maxpool2d(x, /*pool size=*/{2, 2}, /*stride size=*/{2, 2});
        x = op::relu(x);
        x = conv2(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::relu(x);
        x = op::dropout(x, 0.5, is_training);
        x = x.view({x.shape()[0], 7 * 7 * 32});
        x = fc1(x);
        x = op::relu(x);
        x = fc2(x);
        return x;
    }
};
```
You can access the trainable variables of your module.  
```c++
mnist_conv_module model;
model.trainable_variables(); // returns std::vector<tensor>
```

## Deep CNN for CIFAR10

We build a conv block module:  
input -> conv_bn_relu -> dropout -> conv_bn_relu -> maxpool -> dropout.  
Then use it to build deep conv net.  
input -> conv_block -> conv_block -> conv_block -> global avg pool -> fc1 -> fc2.  
See [cifar10 example](example/cifar_v2).  
**The accuracy of this model on cifar10 dataset is about 86% at 30 epoch.**  

```c++
struct conv_block : nn::module{
    nn::conv2d conv1;
    nn::batch_norm2d bn1;
    nn::conv2d conv2;
    nn::batch_norm2d bn2;

    conv_block(){}

    conv_block(int32_t in_chann, int32_t out_chann){
        conv1 = trainable(nn::conv2d(in_chann, out_chann,
            /*kernel_size=*/{3, 3}, /*strides=*/{1, 1},
            /*same_out=*/true, /*use_bias=*/false));
        bn1 = trainable(nn::batch_norm2d(out_chann));
        conv2 = trainable(nn::conv2d(out_chann, out_chann,{3, 3}, {1, 1}, true, false));
        bn2 = trainable(nn::batch_norm2d(out_chann));
    }

    Tensor operator()(Tensor x, float drop_rate, bool is_training=false){
        x = conv1(x);
        x = bn1(x, is_training);
        x = op::relu(x);
        x = conv2(x);
        x = bn2(x, is_training);
        x = op::relu(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::dropout(x, drop_rate, is_training);
        return x;
    }
};

struct cifar10_convnet : nn::module{
    conv_block block1;
    conv_block block2;
    conv_block block3;
    nn::linear fc1;
    nn::batch_norm1d bn1;
    nn::linear fc2;

    cifar10_convnet(){
        block1 = trainable(conv_block(3, 64));
        block2 = trainable(conv_block(64, 128));
        block3 = trainable(conv_block(128, 256));
        fc1 = trainable(nn::linear(4*4*256, 512, /*use_bias=*/false));
        bn1 = trainable(nn::batch_norm1d(512));
        fc2 = trainable(nn::linear(512, 10));
    }

    Tensor forward(Tensor x, bool is_training=false){
        x = block1(x, 0.3, is_training);
        x = block2(x, 0.4, is_training);
        x = block3(x, 0.5, is_training);
        x = x.view({x.shape()[0], 4*4*256});
        x = fc1(x);
        x = bn1(x, is_training);
        x = op::relu(x);
        x = op::dropout(x, 0.5, is_training);
        x = fc2(x);
        return x;
    }

    Tensor criterion(Tensor y_true, Tensor y_pred)
    {
        auto loss = op::softmax_cross_entropy(y_true, y_pred);
        loss = op::reduce_mean(loss);
        return loss;
    }
};
```

## The easiest way to build a model
You don't have to specify the input size of each layer.  
Just stack layers using only output specification using `operator<<()`.  
After stacking sequence layers, call the member function `build` with preferred input shape.  
See [example/imagenet/vgg16.h](example/imagenet/vgg16.h).  

```c++
#include <mlfe/nn/module.h>
#include <mlfe/nn/sequences/batch_norm.h>
#include <mlfe/nn/sequences/conv2d.h>
#include <mlfe/nn/sequences/flatten.h>
#include <mlfe/nn/sequences/maxpool2d.h>
#include <mlfe/nn/sequences/linear.h>
#include <mlfe/nn/sequences/relu.h>

using namespace mlfe;
namespace seq = mlfe::nn::seq;

struct vgg16 : nn::module{
    nn::module net_block;

    template <int C>
    nn::module conv_block(){
        nn::module m;
        return m
            << seq::conv2d<C, size<3, 3>, size<1, 1>, true>()
            << seq::batch_norm2d<>() << seq::relu<>();
    }

    vgg16(){
        net_block
            << conv_block<64>() << conv_block<64>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<128>() << conv_block<128>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<256>() << conv_block<256>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<512>() << conv_block<512>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<512>() << conv_block<512>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << seq::flatten<>()

            << seq::linear<4096>() << seq::batch_norm1d<>() << seq::relu<>()
            << seq::linear<4096>() << seq::batch_norm1d<>() << seq::relu<>()
            << seq::linear<1000>();
        net_block.build({224, 224, 3});
        net_block = trainable(net_block);
    }

    Tensor forward(Tensor x, bool is_train){
        x = net_block(x, is_train);
        return x;
    }
};
```

## How to compile

```
git clone https://github.com/shi510/mlfe
mkdir build
cd build
```
We recomannd to use CUDNN v7.  
See [docker/cudnn_v7_ubuntu18/Dockerfile](docker/cudnn_v7_ubuntu18/Dockerfile).  
```
cmake -D BUILD_TEST=ON -D BUILD_EXAMPLE=ON -D USE_CUDNN=ON -D CMAKE_BUILD_TYPE=Release ..
make -j
./unit_test/unit_test
```
There are another possible options and it is not recommanded.  
```python
-D USE_CUDA=ON # only use cuda kernel not cudnn
-D USE_XNNPACK=ON # currently for experiment
-D USE_INTEL_MKLDNN=ON # currently for experiment
```
It is compiled with host reference codes, if it has no option.  
See [mlfe/operators/impl/cpu](mlfe/operators/impl/cpu).  

## Supported operators
All operators that didn't marked with `'o'` (not implemented yet) will be supported as soon as possible.  

|                       | CUDA(CUDNN) |
|:---------------------:|:-----------:|
|  Add (with broadcast) |      o      |
|  Sub (with broadcast) |      o      |
|  Mul (with broadcast) |      o      |
|  Div (with broadcast) |      o      |
|   GlobalAveragePool2D |      o      |
|           BatchNorm2D |      o      |
|                Conv2D |      o      |
|                 Dense |      o      |
|               Dropout |      o      |
|             MaxPool2D |      o      |
|                Resize |             |
|                  ReLU |      o      |
|               Sigmoid |      o      |
| Softmax Cross Entropy |      o      |
| Sigmoid Cross Entropy |      o      |
|    Squared Difference |      o      |
|             Transpose |             |
|         SGD Optimizer |      o      |
|       Adam Optimizaer |      o      |