#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/operators.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>
#include <numeric>

using namespace mlfe;
namespace fn = functional;

TEST(unary_op, negative){
    using T = float;
    auto var = fn::create_variable({2, 2});
    auto result = fn::negative(var);
    var.mutable_data<T>()[0] = -1.35;
    var.mutable_data<T>()[1] = 357.5;
    var.mutable_data<T>()[2] = -3.5;
    var.mutable_data<T>()[3] = 3.15;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], -var.data<T>()[0]);
    EXPECT_EQ(result.data<T>()[1], -var.data<T>()[1]);
    EXPECT_EQ(result.data<T>()[2], -var.data<T>()[2]);
    EXPECT_EQ(result.data<T>()[3], -var.data<T>()[3]);
}

TEST(unary_op, negative_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto var = fn::create_variable({2, 2});
    auto result = fn::negative(var);
    auto analytical = std::vector<T>(var.size());
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);
    
    std::generate(var.begin<T>(), var.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    result.eval();
    result.backprop();
    var.get_backprop_node().run();
    std::copy(var.grad().begin<T>(), var.grad().end<T>(), analytical.begin());
    auto numerical = numerical_gradient(grad_eps, result, var);
    
    for(int n = 0; n < var.size(); ++n){
        auto diff = std::abs(analytical.data()[n] - numerical.data<T>()[n]);
        EXPECT_LE(diff, pass_eps);
        EXPECT_GE(diff, -pass_eps);
    }
}

TEST(unary_op, sigmoid){
    using T = float;
    auto var = fn::create_variable({2, 2});
    auto result = fn::sigmoid(var);
    var.mutable_data<T>()[0] = 0;
    var.mutable_data<T>()[1] = -0.5;
    var.mutable_data<T>()[2] = 0.5;
    var.mutable_data<T>()[3] = 1;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], 0.5);

    EXPECT_LE(result.data<T>()[1], 0.3775 + 1e-4);
    EXPECT_GE(result.data<T>()[1], 0.3775 - 1e-4);

    EXPECT_LE(result.data<T>()[2], 0.6225 + 1e-4);
    EXPECT_GE(result.data<T>()[2], 0.6225 - 1e-4);

    EXPECT_LE(result.data<T>()[3], 0.7311 + 1e-4);
    EXPECT_GE(result.data<T>()[3], 0.7311 - 1e-4);
}

TEST(unary_op, sigmoid_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto var = fn::create_variable({2, 2});
    auto result = fn::sigmoid(var);
    auto analytical = std::vector<T>(var.size());
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);
    rng.seed(std::random_device()());
    
    std::generate(var.begin<T>(), var.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    result.eval();
    result.backprop();
    var.get_backprop_node().run();
    std::copy(var.grad().begin<T>(), var.grad().end<T>(), analytical.begin());
    auto numerical = numerical_gradient(grad_eps, result, var);
    
    for(int n = 0; n < var.size(); ++n){
        auto diff = std::abs(analytical.data()[n] - numerical.data<T>()[n]);
        EXPECT_LE(diff, pass_eps);
        EXPECT_GE(diff, -pass_eps);
    }
}

TEST(unary_op, relu){
    using T = float;
    auto var = fn::create_variable({2, 2});
    auto result = fn::relu(var);
    var.mutable_data<T>()[0] = -1.35;
    var.mutable_data<T>()[1] = 357.5;
    var.mutable_data<T>()[2] = -3.5;
    var.mutable_data<T>()[3] = 3.15;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], 0);
    EXPECT_EQ(result.data<T>()[1], var.data<T>()[1]);
    EXPECT_EQ(result.data<T>()[2], 0);
    EXPECT_EQ(result.data<T>()[3], var.data<T>()[3]);
}

TEST(unary_op, relu_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto var = fn::create_variable({2, 2});
    auto result = fn::relu(var);
    auto analytical = std::vector<T>(var.size());
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);
    
    std::generate(var.begin<T>(), var.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    result.eval();
    result.backprop();
    var.get_backprop_node().run();
    std::copy(var.grad().begin<T>(), var.grad().end<T>(), analytical.begin());
    auto numerical = numerical_gradient(grad_eps, result, var);
    
    for(int n = 0; n < var.size(); ++n){
        auto diff = std::abs(analytical.data()[n] - numerical.data<T>()[n]);
        EXPECT_LE(diff, pass_eps);
        EXPECT_GE(diff, -pass_eps);
    }
}

TEST(unary_op, mean) {
    using T = float;
    auto x = fn::create_variable({ 5 });
    auto y = fn::mean(x);
    T answer = 0.f;
    std::fill(x.begin<float>(), x.end<float>(), 0.2f);
    for (int n = 0; n < x.size(); ++n)
    {
        answer += 0.2f;
    }
    answer /= x.size();
    y.eval();
    EXPECT_EQ(y.size(), 1);
    EXPECT_EQ(y.data<T>()[0], answer);
}

TEST(unary_op, mean_grad) {
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto x = fn::create_variable({ 5 });
    auto y = fn::mean(x);
    auto analytical = std::vector<T>(x.size());
    std::mt19937 rng;
    std::normal_distribution<T> dist(-1, 1);

    std::generate(x.begin<T>(), x.end<T>(), [&]() {return dist(rng);});
    y.eval();
    y.backprop();
    x.get_backprop_node().run();
    std::copy(x.grad().begin<T>(), x.grad().end<T>(), analytical.begin());
    auto numerical = numerical_gradient(grad_eps, y, x);

    for (int n = 0; n < x.size(); ++n) {
        //auto diff = std::abs(analytical.data()[n] - numerical.data<T>()[n]);
        //std::cout<< analytical.data()[n] <<", "<< numerical.data<T>()[n] <<std::endl;
        //EXPECT_LE(diff, pass_eps);
        //EXPECT_GE(diff, -pass_eps);
        EXPECT_NEAR(analytical.data()[n], numerical.data<T>()[n], pass_eps);
    }
}
