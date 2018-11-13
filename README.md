
# mlfe : Modeling Learnable Feature Extractor  
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/shi510/mlfe.svg?branch=master)](https://travis-ci.org/shi510/mlfe)  

MLFE is a framework for machine learning written by modern C++.  
Initially this project was for studying on backpropagation algorithm of deep learning, but we decided to develop deeper for mobile platform.  
So, our goal is to optimize a neural network on mobile platform by quantizing or compressing a neural network.  
Currently, it supports C++ lang, but we are working now to support Rust lang and Go lang.  

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

auto one = fn::create_variable({2, 2});
auto two = fn::constant(2, {2, 2});
std::fill(one.begin<float>(), one.end<float>(), 1);
// same result with std::fill.
// acceessing address pointer directly.
for(int n = 0; n < one.size(); ++n){
    one.mutable_data<float>()[n] = 1;
}
```
Here, we build simple operations that is 3 * (x + 2)^2 and then we apply mean function.  
```c++
auto three = one + two;
auto y = three * three * fn::constant(3, three.shape());
auto result = fn::mean(y);
```

You can get the result by calling eval function and the value is 27.  
All gradients of variables can calculate by calling backprop function and can access by using grad function.  
The gradient of one is 4.5.  
```c++
result.eval();
result.backprop();
one.grad();
```

## Simple Neural Network for MNIST Dataset.

To train mnist data, we build a simple neural network.  
This code is in example/train/mnist_train.cc.

*mnist data -> fully connected NN -> softmax.*

First step is to including headers.
```c++
#include <mlfe/core.h>
#include <mlfe/operators.h>
#include <mlfe/optimizer.h>
#include <vector>
```
For convenience, we use namespace abbreviation.
```c++
using namespace mlfe;
namespace fn = functional;
```

The mini-batch size is 64 and the mnist image width and height are 28 and 28, respectively.  
Our model input shape is [64, 784] and output shape is [64, 10].  
```c++
int main(int argc, char *argv[]){
    using T = float;
    constexpr int batch = 64;
    constexpr int cls = 10;
    constexpr int h = 28;
    constexpr int w = 28;
    constexpr int iter = 1000;
    auto x = fn::create_variable({batch, h * w});
    auto onehot =  fn::create_variable({batch, cls});
```
The fully connected NN can be implemented by using matrix multiplication.  
The cost function is softmax cross entropy.  
The trainable variables of w and b are optimized by gradient descent method.  
We set the learning rate to 1e-1.  
Initializing the trainable variables is important, but in this example, we just initialize w and b to zero.  
```c++
    auto w = fn::create_variable({h * w, cls});
    auto b = fn::create_variable({cls});
    auto logit = fn::matmul(x, w) + b;
    auto loss = fn::softmax_cross_entropy(logit, onehot);
    auto sgd_opt = fn::create_gradient_descent(/* learning rate = */ 0.1,
                                               /* momentum = */ 0);
    std::fill(w.begin<T>(), w.end<T>(), 0);
    std::fill(b.begin<T>(), b.end<T>(), 0);
```

An original mnist file can convert into simpledb and can read using simpledb reader.  
The simpledb is NoSQL key-value DB that we developed.
```c++
    auto train_db = SimbleDBReader(argv[1]);
    auto img = std::vector<T>(batch * h * w)
    auto label = std::vector<T>(batch);
```

First, we should transform the input data value to [0, 1).  
the range of mnist data value is [0, 255] and we divide by 256.  
Second, we should transform the label data to one-hot form.  
If the label is 5, the one-hot form represents [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].  
```c++
    for(int n = 0; n < iter; ++n){
        train_db.read<T>(batch, {img, label});
        for(auto &val : img){
            val /= 256;
        }
        std::copy(img.begin(), img.end(), x.begin<T>());
        std::fill(onehot.begin<T>(), onehot.end<T>(), 0);
        for(int k = 0; k < batch; ++k){
            const int val = label.data()[k];
            onehot.mutable_data<T>()[k * cls + val] = 1;
        }
        loss.eval();
        loss.backprop();
        sgd_opt.apply(w, w.grad());
        sgd_opt.apply(b, b.grad());
    }
```
After 1000 iterations are finished, the variables will close to the optimal solution.  
This model performs about 90% acccuracy on MNIST 10K test images.  
The following figure shows visualization of the w variable, the Red colour represents negative value, the Blue colour represents positive value.

![visualization weights](http://artoa.hanbat.ac.kr/simple_mnist_weights.jpg)
