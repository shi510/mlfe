
# mlfe : Modeling Learnable Feature Extractor  
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/shi510/mlfe.svg?branch=master)](https://travis-ci.org/shi510/mlfe)  

MLFE is a framework for machine learning written by modern C++.  
Initially this project was for studying on backpropagation algorithm of deep learning, but we decided to develop deeper for mobile platform.  
So, our goal is to optimize a neural network on mobile platform by quantizing or compressing a neural network.  
It supports only C++ lang now, but we are working to support Rust lang and Go lang.  

## API Proposals For New Architecture
The API described below is candidates for new architecture.  
It is almost finalized and the released version may have tiny changes.  

## Index
1. [Basic Example](#Basic-Example)
2. [Simple Neural Network for MNIST Dataset](#Simple-neural-network-for-MNIST-dataset.)
3. [Convolutional neural network](#Convolutional-neural-network)
4. [The easiest way to build a model](#The-easiest-way-to-build-a-model)
5. [Supported operators](#Supported-operators)

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
#include <mlfe/operators_v2/reduce_mean.h>
#include <mlfe/operators_v2/softmax_cross_entropy.h>
#include <mlfe/operators_v2/matmul.h>
```
For convenience, we use namespace abbreviation.  
```c++
using namespace mlfe;
namespace fn = functional;
namespace op = operators_v2;
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

## Convolutional neural network

1. Inherit the nn::module.  
2. Create layers in your class.  
3. Notify trainable variables to nn::module by enclosing a layer with `trainable` function.  

The `trainable` function finds trainables in a layer and collects it.  

```c++
using namespace mlfe;
namespace op = mlfe::operators_v2;

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
                                   /*output channel=*/512));
        fc2 = trainable(nn::linear(512, 10));
    }

    tensor forward(tensor x){
        x = conv1(x);
        x = op::relu(x);
        x = op::maxpool2d(x, /*pool size=*/{2, 2}, /*stride size=*/{2, 2});
        x = conv2(x);
        x = op::relu(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = x.view({x.shape()[0], 7*7*32});
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

## The easiest way to build a model
**`In Progress Now.`**  
Just push layers using `operator<<()` to a nn::module, then call it.  
You don't have to specify the input size of a layer, such as nn::linear(`784`, 300).  

```c++
struct vgg16 : nn::module{

    template <int C>
    auto conv_block(){
        return nn::module()
            << seq::conv2d<C, size<3, 3>, size<1, 1>, true>()
            << seq::batch_norm<>() << seq::relu<>();
    }

    vgg16(){
        net_block
            << conv_block<32>() << conv_block<32>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<64>() << conv_block<64>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<128>() << conv_block<128>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<256>() << conv_block<256>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<512>() << conv_block<512>() << conv_block<512>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<512>() << conv_block<512>() << conv_block<512>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << seq::linear<4096>() << seq::relu<>()
            << seq::linear<4096>() << seq::relu<>()
            << seq::linear<1000>();
        net_block.build({224, 224, 3});
        trainable(net_block);
    }

    tensor forward(tensor x){
        x = net_block(x)
        return x;
    }

    nn::module net_block;
};
```

## Supported operators
All operators that didn't marked with `'o'` (not implemented yet) will be supported as soon as possible.  

|                       | ARMv8 | CUDA(CUDNN) | x86_64 |
|:---------------------:|:-----:|:-----------:|:------:|
|         AveragePool2D |       |             |        |
|           BatchNorm2D |       |      o      |        |
|                Conv2D |   o   |      o      |    o   |
|                 Dense |   o   |      o      |    o   |
|               Dropout |       |             |        |
|             MaxPool2D |   o   |      o      |    o   |
|                Resize |       |             |        |
|                  ReLU |   o   |      o      |    o   |
|               Sigmoid |   o   |      o      |    o   |
| Softmax Cross Entropy |   o   |      o      |    o   |
| Sigmoid Cross Entropy |   o   |      o      |    o   |
|    Squared Difference |   o   |      o      |    o   |
|             Transpose |       |             |        |
|         SGD Optimizer |   o   |      o      |    o   |
|       Adam Optimizaer |   o   |      o      |    o   |
