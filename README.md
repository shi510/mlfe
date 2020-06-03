
# mlfe : Modeling Learnable Feature Extractor  
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/shi510/mlfe.svg?branch=master)](https://travis-ci.org/shi510/mlfe)  

MLFE is a framework for machine learning written by modern C++.  
Initially this project was for studying on backpropagation algorithm of deep learning, but we decided to develop deeper for mobile platform.  
So, our goal is to optimize a neural network on mobile platform by quantizing or compressing a neural network.  
It supports only C++ lang now, but we are working to support Rust lang and Go lang.  

## API is not stable.  
Note that API is not stable currently, it will be removed or modified without notifications.  
See examples or unit-test implementations for details.  
Those will be updated along with the API changes.  

## Keras-like API
Keras API (TensorFlow) makes you build a neural network easily.  
MLFE adopts Keras-like API.  
To train MNIST dataset using Keras-like API, build a network first as below.  
```c++
namespace models
{
using namespace mlfe::module;
using namespace mlfe::module::layers;

model conv_net(std::vector<int> input_shape)
{
    auto in = input(input_shape)();
    auto out = conv2d(16, 5, 1, true)(in);
    out = maxpool2d(2, 2, 0)(out);
    out = relu()(out);
    out = conv2d(24, 5, 1, true)(out);
    out = maxpool2d(2, 2, 0)(out);
    out = relu()(out);
    out = flatten()(out);
    out = dense(128)(out);
    out = relu()(out);
    out = dense(10)(out);
    return model(in, out);
}

} // end namespace models
```

Secondly, prepare your MNIST dataset.  
It is implemented in examples/mnist/dataset/mnist.cc.  
```c++
std::vector<uint8_t> train_x; // 60000 train images.
std::vector<uint8_t> train_y; // 60000 train labels. 
std::vector<uint8_t> valid_x; // 10000 test images.
std::vector<uint8_t> valid_y; // 10000 test labels.
read_mnist_dataset("MNIST data path", train_x, train_y, valid_x, valid_y);
```

Thirdly, implement custom Generator class.  
See mnist_gen class in examples/mnist/dataset/mnist.h.  
The mnist_gen class is callable and returns a tuple by operator(int idx).  
```c++
dataset::mnist_gen train_set(train_x, train_y), valid_set(valid_x, valid_y);
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> data_label = train_set(0);
```

Lastly, choose your optimizer and loss function.  
Then, train your model by calling fit member function.  
After 5 epochs, test-set accuracy is about 99%.  
See examples/mnist/main.cc for more details.  
```c++
constexpr int B = 64;
constexpr int EPOCHS = 5;
auto net = models::conv_net({B, 1, 28, 28});
auto optm = functional::create_gradient_descent_optimizer(2e-2, 0.9);
auto loss = functional::softmax_cross_entropy;
net.compile(optm, loss, categorical_accuracy);
net.fit(train_set, valid_set, EPOCHS, B);
```
## Tensorboard
MLFE supports tensorboard.  
You can customize callback for tensorboard.  
```c++
class custom_histo_weights : public callback
{
public:
    custom_histo_weights(std::string log_dir)
    {
        __writer = std::make_shared<util::summary_writer>(log_dir + "/hist/tfevents.pb");
    }

    void on_epoch_end(const int epoch,
        const std::map<std::string, float>& logs) override
    {
        for (auto& var : __m->get_train_variables())
        {
            auto pos1 = var.name().find("dense");
            auto pos2 = var.name().find("weights");
            if (pos1 != std::string::npos && pos2 != std::string::npos)
            {
                std::vector<float> w(var.size());
                std::copy(var.cbegin<float>(), var.cend<float>(), w.begin());
                __writer->add_histogram(var.name(), epoch, w);
            }
        }
    }

private:
    std::shared_ptr<util::summary_writer> __writer;
};
```
The model class will call on_epoch_end function at every end of epoch.  
See the example in examples/cifar/main.cc for more details.  
```c++
net.fit(train_set, valid_set, 100, _BatchSize,
    { reduce_lr("valid/loss", 3), tensorboard("cifar10_logs"),
        custom_histo_weights("cifar10_logs") });
```
![tensorboard_scalar](https://raw.githubusercontent.com/shi510/mlfe/master/figures/fig_tensorboard_scalar.jpg)
![tensorboard_histogram](https://raw.githubusercontent.com/shi510/mlfe/master/figures/fig_tensorboard_histo.jpg)

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
// accessing address pointer directly.
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

## Simple Neural Network for MNIST Dataset using low-level API.

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
The trainable variables of weight and bias are optimized by gradient descent method.  
The first parameter of create_gradient_descent_optimizer is learning rate and the last is momentum.  
Initializing the trainable variable is important.  
But in this example, we just initialize weight and bias to zero.  
```c++
    auto weight = fn::create_variable({h * w, cls});
    auto bias = fn::create_variable({cls});
    auto logit = fn::matmul(x, weight) + bias;
    auto loss = fn::softmax_cross_entropy(logit, onehot);
    auto sgd_opt = fn::create_gradient_descent_optimizer(1e-1, 0);
    std::fill(weight.begin<T>(), weight.end<T>(), 0);
    std::fill(bias.begin<T>(), bias.end<T>(), 0);
```

Create std vectors to store image and label data batch.
```c++
    auto img = std::vector<T>(batch * h * w)
    auto label = std::vector<T>(batch);
```

Normalize the image data value to [0, 1].  
The range of image data is [0, 255], so it is divided by 255.  
Transform the label data to one-hot form.  
If the label is 5, the one-hot form represents [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].  
```c++
    for(int n = 0; n < iter; ++n){
        // read mnist data.
        // implement read_mnist_data_batch and read_mnist_label_batch.
        read_mnist_data_batch(img);
        read_mnist_label_batch(label);
        // normalize to [0, 1).
        for(auto &val : img){
            val /= 255.f;
        }
        // copy normalized host data to our model's input x.
        std::copy(img.begin(), img.end(), x.begin<T>());
        // initialize onehot vector to zero.
        std::fill(onehot.begin<T>(), onehot.end<T>(), 0);
        // transform label to onehot form.
        for(int k = 0; k < batch; ++k){
            const int val = label.data()[k];
            onehot.mutable_data<T>()[k * cls + val] = 1;
        }
        // evaluate loss.
        loss.eval();
        // evaluate all gradients associated with loss variable.
        loss.backprop();
        // update weight -= lr * gradient of weight.
        sgd_opt.apply(weight, weight.grad());
        // update weight -= lr * gradient of bias.
        sgd_opt.apply(bias, bias.grad());
    }
```
After 1000 iterations are finished, the variables will close to the optimal solution.  
This model performs about 90% acccuracy on MNIST 10K test images.  
The following figure shows visualization of the w variable, the Red colour represents negative value, the Blue colour represents positive value.

<!-- ![visualization weights](http://artoa.hanbat.ac.kr/simple_mnist_weights.jpg) -->
![visualization weights](https://raw.githubusercontent.com/shi510/mlfe/master/figures/fig_mnist_weights.jpg)
