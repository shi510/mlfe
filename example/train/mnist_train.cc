#include "mnist_train.h"
#include "mnist_models.h"
#include <mlfe/core.h>
#include <mlfe/operators.h>
#include <mlfe/optimizers.h>
#include <mlfe/utils/db/simpledb_reader.h>
#include <opencv2/opencv.hpp>
#include <random>

namespace train_example{
using namespace mlfe;
namespace fn = functional;

void Visualize(cv::Mat *img, std::vector<float> decode){
    const int classes = 10;
    const int size = 28;
    for(int n = 0; n < classes; ++n) {
        cv::Mat cv_decode(size, size, CV_32FC1, decode.data() + n * size * size);
        for(int r = 0; r < size; ++r) {
            for(int c = 0; c < size; ++c) {
                float val = cv_decode.at<float>(r, c);
                img->at<float>(r, (c + n * size)) = cv_decode.at<float>(r, c);
            }
        }
    }
}

void train_simple_mnist(const std::string train_path, 
                        const std::string test_path, 
                        const int batch,
                        const int iter, 
                        const double lr,
                        const double mm
                       ){
    constexpr int cls = 10;
    Tensor x = fn::create_variable({batch, 28 * 28});
    Tensor y = fn::create_variable({batch, cls});
    auto sgd = fn::create_gradient_descent_optimizer(lr, mm);
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(0.1);

    std::vector<float> x_val, y_val;
    x_val.resize(x.size());
    y_val.resize(batch);
    auto train_db = mlfe::SimpleDBReader(train_path);
    auto test_db = mlfe::SimpleDBReader(test_path);

    Tensor w = functional::create_variable({28 * 28, cls});
    Tensor b = functional::create_variable({cls});
    std::fill(w.mutable_data<float>(), 
              w.mutable_data<float>() + w.size(), 
              0);
    std::fill(b.mutable_data<float>(), 
              b.mutable_data<float>() + b.size(), 
              0);

    auto out = fn::add(fn::matmul(x, w), b);
    auto loss = fn::mean(fn::softmax_cross_entropy(out, y));
    for(int n = 0; n < iter; ++n) {
        std::vector<float> onehot(batch * cls);
        std::fill(onehot.begin(), onehot.end(), 0);
        train_db.Read<float>(batch, {x_val, y_val});
        for(auto &val : x_val){ val = val / 255.f; }
        for(int n = 0; n < batch; ++n){
            onehot.data()[(int)y_val.data()[n] + n * cls] = 1.f;
        }
        std::copy(x_val.begin(), x_val.end(), x.begin<float>());
        std::copy(onehot.begin(), onehot.end(), y.begin<float>());
        
        loss.eval();
        loss.backprop();
        sgd->apply(w, w.grad());
        sgd->apply(b, b.grad());
        
        if(((n + 1) % 100) == 0) {
            int test_iter = 10000. / float(64) + 0.5;
            int corrent = 0;
            for(int m = 0; m < test_iter; ++m){
                std::vector<float> out_val;
                std::fill(onehot.begin(), onehot.end(), 0.f);
                test_db.Read<float>(batch, { x_val, y_val });
                for(auto &val : x_val){
                    val /= 256.f;
                }
                for(int k = 0; k < batch; ++k){
                    onehot.data()[(int)y_val.data()[k] + k * cls] = 1.f;
                }
                std::copy(x_val.begin(), x_val.end(), x.begin<float>());
                std::copy(onehot.begin(), onehot.end(), y.begin<float>());
                out.eval();
                out_val.assign(out.data<float>(), out.data<float>() + out.size());
                for(int k = 0; k < batch; ++k){
                    auto s = out_val.begin() + k * cls;
                    auto e = out_val.begin() + (k + 1) * cls;
                    auto pos = std::max_element(s, e);
                    int infer = std::distance(out_val.begin() + k * cls, pos);
                    if(y_val.data()[k] == infer){
                        corrent += 1;
                    }
                }
            }
            std::cout << "Iter : " << n + 1 << " : " << std::endl;
            std::cout << "  Accracy over test 10K images : ";
            std::cout << double(corrent) / (test_iter * batch) << std::endl;
        }
    }
}

void train_lenet(const std::string train_path,
                 const std::string test_path,
                 const int batch,
                 const int iter,
                 const double lr,
                 const double mm
                ){
    Lenet lenet(batch, lr, mm);
    std::vector<float> onehot;
    std::vector<float> x, y;
    x.resize(batch * 28 * 28);
    y.resize(batch);
    onehot.resize(batch * 10);

    auto train_db = mlfe::SimpleDBReader(train_path);
    auto test_db = mlfe::SimpleDBReader(test_path);
    
    for(int n = 0; n < iter; ++n) {
        std::fill(onehot.begin(), onehot.end(), 0);
        train_db.Read<float>(batch, {x, y});
        for(auto &val : x){ val = val / 255.f; }
        for(int k = 0; k < batch; ++k){
            onehot.data()[(int)y.data()[k] + k * 10] = 1.f;
        }
        lenet.forward(x, onehot);
        lenet.backward();
        lenet.update();
        if(((n + 1) % 500) == 0) {
            int test_iter = 10000. / float(batch) + 0.5;
            double loss_mean = 0.;
            int corrent = 0;
            test_db.MoveToFirst();
            for(int m = 0; m < test_iter; ++m){
                std::fill(onehot.begin(), onehot.end(), 0);
                test_db.Read<float>(batch, {x, y});
                for(auto &val : x){ val = val / 255.f; }
                for(int k = 0; k < batch; ++k){
                    onehot.data()[(int)y.data()[k] + k * 10] = 1.f;
                }
                auto result = lenet.get_logit_loss(x, onehot);
                auto logit = std::get<0>(result);
                auto loss = std::get<1>(result);
                loss_mean += loss[0];
                for(int b = 0; b < batch; ++b){
                    auto s = logit.begin() + b * 10;
                    auto e = logit.begin() + (b + 1) * 10;
                    auto pos = std::max_element(s, e);
                    int infer = std::distance(s, pos);
                    if(y.data()[b] == infer){
                        corrent += 1;
                    }
                }
            }
            std::cout << "Iter : " << n + 1 << " : " << std::endl;
            std::cout << "     Loss over test 10K images : ";
            std::cout << loss_mean / test_iter << std::endl;
            std::cout << "  Accracy over test 10K images : ";
            std::cout << double(corrent) / (test_iter * batch) << std::endl;
        }
    }
}

void train_ae(const std::string train_path,
              const std::string test_path,
              const int batch,
              const int iter,
              const double lr,
              const double mm
             ){
    AutoEncoder ae(batch, lr, mm);
    cv::Mat visual(28, 28 * 10, CV_32FC1);
    cv::Mat origin(28, 28 * 10, CV_32FC1);
    cv::Mat resized, origin_resized;
    std::vector<float> input, label;
    std::vector<float> input_test;
    input.resize(batch * 28 * 28);
    label.resize(batch);
    input_test.resize(batch * 28 * 28);
    auto train_db = mlfe::SimpleDBReader(train_path);
    auto test_db = mlfe::SimpleDBReader(test_path);
    test_db.Read<float>(batch, { input_test, label });
    for(auto &val : input_test){ val = val / 255.f; }
    test_db.MoveToFirst();

    for(int n = 0; n < iter; ++n) {
        train_db.Read<float>(batch, { input, label });
        for(auto &val : input){ val = val / 255.f; }
        ae.forward(input);
        ae.backward();
        ae.update();
        if(((n + 1) % 50) == 0) {
            auto recon_val = ae.recon(input_test);
            visual = 0;
            origin = 0;
            Visualize(&visual, recon_val);
            Visualize(&origin, input_test);
            cv::resize(visual, resized, visual.size() * 4);
            cv::resize(origin, origin_resized, origin.size() * 4);
            cv::imshow("original", origin_resized);
            cv::imshow("reconstruction", resized);
            cv::waitKey(1);
        }
        if(((n + 1) % 1000) == 0) {
            int test_iter = 10000. / batch + 0.5;
            double loss_mean = 0.;
            for(int n = 0; n < test_iter; ++n){
                test_db.Read<float>(batch, { input, label });
                for(auto &val : input){ val = val / 255.f; }
                auto loss_val = ae.get_loss(input);
                loss_mean += loss_val[0];
            }
            std::cout << n + 1 << " - loss : ";
            std::cout << loss_mean / test_iter << std::endl;
            visual = 0;
        }
    }
    cv::waitKey(0);
}
} // end namespace train_example