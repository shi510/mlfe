#ifndef __EXAMPLE_MNIST_MODELS_H__
#define __EXAMPLE_MNIST_MODELS_H__
#include <mlfe/core/tensor.h>
#include <random>
#include <unordered_map>
#include <mlfe/optimizers.h>

namespace train_example{

struct AutoEncoder{
    AutoEncoder(const int batch, const double lr, const double mm);

    std::vector<float> recon(const std::vector<float> &x_val);

    void forward(const std::vector<float> &x_val);

    void backward();

    std::vector<float> get_loss(const std::vector<float> &x_val);

    void build(const int batch);

    mlfe::Tensor encoder(mlfe::Tensor x);

    mlfe::Tensor decoder(mlfe::Tensor x);

    mlfe::Tensor fc_relu(const std::string name, 
                         mlfe::Tensor x, 
                         const int out, 
                         const double std,
                         const bool is_act
                        );

    void update();

    mlfe::Tensor x;
    mlfe::Tensor encode;
    mlfe::Tensor decode;
    mlfe::Tensor decode_with_sigmoid;
    mlfe::Tensor loss;
    mlfe::Tensor dropout_prob;
    mlfe::opt::optimizer_ptr sgd;
    std::mt19937 rng;
    std::unordered_map<std::string, mlfe::Tensor> vars;
};

struct Lenet{
    using LogitLoss = std::tuple<std::vector<float>, std::vector<float>>;

    Lenet(const int batch, const double lr, const double mm);

    void forward(const std::vector<float> &x_val,
                 const std::vector<float> &y_val);

    void backward();

    LogitLoss get_logit_loss(const std::vector<float> &x_val,
                             const std::vector<float> &y_val);

    void build(const int batch);

    mlfe::Tensor fc_relu(const std::string name,
                         const mlfe::Tensor x,
                         const int out,
                         const double std,
                         const bool is_act
                        );

    mlfe::Tensor conv2d(const std::string name,
                        const mlfe::Tensor x,
                        const int filter,
                        const int kernel,
                        const int stride,
                        const int padding,
                        const double std
                       );

    mlfe::Tensor maxpool(const std::string name,
                         const mlfe::Tensor x,
                         const int kernel,
                         const int stride,
                         const int padding
                        );

    void update();

    mlfe::Tensor x;
    mlfe::Tensor y;
    mlfe::Tensor logit;
    mlfe::Tensor loss;
    mlfe::opt::optimizer_ptr sgd;
    std::mt19937 rng;
    std::unordered_map<std::string, mlfe::Tensor> vars;
};

} // end namespace train_example
#endif // __EXAMPLE_MNIST_MODELS_H__