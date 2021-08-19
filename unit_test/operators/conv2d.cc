#include <gtest/gtest.h>
#include <mlfe/operators/conv2d.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

// TODO : add more kernel, stride and padding size test.
TEST(operator, conv2d_k3_s1_p0){
    using T = float;
    constexpr T eps = 1e-5;
    constexpr int b = 1;
    constexpr int ci = 2;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int co = 1;
    constexpr int k1 = 3;
    constexpr int k2 = 3;
    auto x = fn::create_variable({b, hi, wi, ci});
    auto w = fn::create_variable({k1, k2, ci, co});
    // x =
    //    [  1  2  3  4    [  1  5  9 13
    //       5  6  7  8       2  6 10 14
    //       9 10 11 12       3  7 11 15
    //      13 14 15 16 ]     4  8 12 16 ]
    for(int r = 0; r < hi; ++r){
        for(int c = 0; c < wi; ++c){
            const int idx = r * wi * ci + c * ci;
            x.mutable_data<T>()[idx] = r * wi + c + 1;
            
        }
    }

    for(int c = 0; c < wi; ++c){
        for(int r = 0; r < hi; ++r){
            const int idx = r * wi * ci + c * ci + 1;
            x.mutable_data<T>()[idx] = r + c * hi + 1;
        }
    }

    // w =
    //    [ 0.1 0.2 0.3    [ 0.1 0.4 0.7
    //      0.4 0.5 0.6      0.2 0.5 0.8
    //      0.7 0.8 0.9 ]    0.3 0.6 0.9 ]
    for(int r = 0; r < k1; ++r){
        for(int c = 0; c < k2; ++c){
            const int idx = r * k2 * ci * co + c * ci * co;
            w.mutable_data<T>()[idx] = (r * k2 + c + 1) * T(1e-1);
        }
    }

    for(int c = 0; c < k2; ++c){
        for(int r = 0; r < k1; ++r){
            const int idx = r * k2 * ci * co + c * ci * co + 1 * co;
            w.mutable_data<T>()[idx] = (r + c * k1 + 1) * T(1e-1);
        }
    }

    auto y = conv2d(x, w, {1, 1}, {0, 0});

    //y(0, 0) = 1 * 0.1 +  2 * 0.2 +  3 * 0.3 +
    //          5 * 0.4 +  6 * 0.5 +  7 * 0.6 +
    //          9 * 0.7 + 10 * 0.8 + 11 * 0.9 +
    //
    //          1 * 0.1 + 5 * 0.4 +  9 * 0.7 +
    //          2 * 0.2 + 6 * 0.5 + 10 * 0.8 +
    //          3 * 0.3 + 7 * 0.6 + 11 * 0.9
    //        = 34.8 + 34.8
    EXPECT_LE(y.data<float>()[0], 69.6 + eps);
    EXPECT_GE(y.data<float>()[0], 69.6 - eps);

    //y(0, 1) =  2 * 0.1 +  3 * 0.2 +  4 * 0.3 +
    //           6 * 0.4 +  7 * 0.5 +  8 * 0.6 +
    //          10 * 0.7 + 11 * 0.8 + 12 * 0.9 +
    //
    //          5 * 0.1 +  9 * 0.4 + 13 * 0.7 +
    //          6 * 0.2 + 10 * 0.5 + 14 * 0.8 +
    //          7 * 0.3 + 11 * 0.6 + 15 * 0.9
    //        = 92.1
    EXPECT_LE(y.data<float>()[1], 92.1 + eps);
    EXPECT_GE(y.data<float>()[1], 92.1 - eps);

    //y(1, 0) =  5 * 0.1 +  6 * 0.2 +  7 * 0.3 +
    //           9 * 0.4 + 10 * 0.5 + 11 * 0.6 +
    //          13 * 0.7 + 14 * 0.8 + 15 * 0.9 +
    //
    //          2 * 0.1 + 6 * 0.4 + 10 * 0.7 +
    //          3 * 0.2 + 7 * 0.5 + 11 * 0.8 +
    //          4 * 0.3 + 8 * 0.6 + 12 * 0.9
    //        = 92.1
    EXPECT_LE(y.data<float>()[2], 92.1 + eps);
    EXPECT_GE(y.data<float>()[2], 92.1 - eps);

    //y(1, 1) =  6 * 0.1 +  7 * 0.2 +  8 * 0.3 +
    //          10 * 0.4 + 11 * 0.5 + 12 * 0.6 +
    //          14 * 0.7 + 15 * 0.8 + 16 * 0.9 +
    //
    //          6 * 0.1 + 10 * 0.4 + 14 * 0.7 +
    //          7 * 0.2 + 11 * 0.5 + 15 * 0.8 +
    //          8 * 0.3 + 12 * 0.6 + 16 * 0.9
    //        = 114.6
    EXPECT_LE(y.data<float>()[3], 114.6 + eps);
    EXPECT_GE(y.data<float>()[3], 114.6 - eps);
}


TEST(operator, conv2d_k3_s2_p1){
    using T = float;
    constexpr T eps = 1e-5;
    constexpr int b = 1;
    constexpr int ci = 2;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int co = 1;
    constexpr int k1 = 3;
    constexpr int k2 = 3;
    auto x = fn::create_variable({b, hi, wi, ci});
    auto w = fn::create_variable({k1, k2, ci, co});
    // x =
    //    [  1  2  3  4    [  1  5  9 13
    //       5  6  7  8       2  6 10 14
    //       9 10 11 12       3  7 11 15
    //      13 14 15 16 ]     4  8 12 16 ]
    for(int r = 0; r < hi; ++r){
        for(int c = 0; c < wi; ++c){
            const int idx = r * wi * ci + c * ci;
            x.mutable_data<T>()[idx] = r * wi + c + 1;
            
        }
    }

    for(int c = 0; c < wi; ++c){
        for(int r = 0; r < hi; ++r){
            const int idx = r * wi * ci + c * ci + 1;
            x.mutable_data<T>()[idx] = r + c * hi + 1;
        }
    }

    // w =
    //    [ 0.1 0.2 0.3    [ 0.1 0.4 0.7
    //      0.4 0.5 0.6      0.2 0.5 0.8
    //      0.7 0.8 0.9 ]    0.3 0.6 0.9 ]
    for(int r = 0; r < k1; ++r){
        for(int c = 0; c < k2; ++c){
            const int idx = r * k2 * ci * co + c * ci * co;
            w.mutable_data<T>()[idx] = (r * k2 + c + 1) * T(1e-1);
        }
    }

    for(int c = 0; c < k2; ++c){
        for(int r = 0; r < k1; ++r){
            const int idx = r * k2 * ci * co + c * ci * co + 1 * co;
            w.mutable_data<T>()[idx] = (r + c * k1 + 1) * T(1e-1);
        }
    }
    auto y = conv2d(x, w, {2, 2}, {1, 1});
    EXPECT_EQ(y.shape()[0], b);
    EXPECT_EQ(y.shape()[1], 2);
    EXPECT_EQ(y.shape()[2], 2);
    EXPECT_EQ(y.shape()[3], co);

    //y(0, 0) = 1 * 0.5 +  2 * 0.6 + 
    //          5 * 0.8 +  6 * 0.9 + 
    //
    //          1 * 0.5 + 5 * 0.8 + 
    //          2 * 0.6 + 6 * 0.9
    //        = 22.2
    EXPECT_NEAR(y.data<float>()[0], 22.2, eps);

    //y(0, 1) =  2 * 0.4 +  3 * 0.5 +  4 * 0.6 +
    //           6 * 0.7 +  7 * 0.8 +  8 * 0.9 +
    //
    //          5 * 0.2 +  9 * 0.5 + 13 * 0.8 +
    //          6 * 0.3 + 10 * 0.6 + 14 * 0.9
    //        = 58
    EXPECT_NEAR(y.data<float>()[1], 58, eps);

    //y(1, 0) =  5 * 0.2 +  6 * 0.3 +
    //           9 * 0.5 + 10 * 0.6 +
    //          13 * 0.8 + 14 * 0.9 +
    //
    //          2 * 0.4 + 6 * 0.7 +
    //          3 * 0.5 + 7 * 0.8 +
    //          4 * 0.6 + 8 * 0.9
    //        = 58
    EXPECT_NEAR(y.data<float>()[2], 58, eps);

    //y(1, 1) =  6 * 0.1 +  7 * 0.2 +  8 * 0.3 +
    //          10 * 0.4 + 11 * 0.5 + 12 * 0.6 +
    //          14 * 0.7 + 15 * 0.8 + 16 * 0.9 +
    //
    //          6 * 0.1 + 10 * 0.4 + 14 * 0.7 +
    //          7 * 0.2 + 11 * 0.5 + 15 * 0.8 +
    //          8 * 0.3 + 12 * 0.6 + 16 * 0.9
    //        = 114.6
    EXPECT_NEAR(y.data<float>()[3], 114.6, eps);
}

TEST(operator, conv2d_k3_s1_p0_grad){  
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    constexpr int b = 1;
    constexpr int ci = 1;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int co = 1;
    constexpr int k1 = 3;
    constexpr int k2 = 3;
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);
    auto x = fn::create_variable({b, hi, wi, ci});
    auto w = fn::create_variable({k1, k2, ci, co});
    auto analytical_x = fn::create_variable(x.shape());
    auto analytical_w = fn::create_variable(w.shape());

    std::generate(x.begin<T>(), x.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(w.begin<T>(), w.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto y = conv2d(x, w, {1, 1}, {0, 0});
    y.backprop();
    std::copy(x.grad().cbegin<T>(), x.grad().cend<T>(),
        analytical_x.begin<T>());
    std::copy(w.grad().cbegin<T>(), w.grad().cend<T>(),
        analytical_w.begin<T>());
    auto func1 = [w](mlfe::Tensor& x){
        return conv2d(x, w, {1, 1}, {0, 0});
    };
    auto numerical_x = numerical_gradient_v2(func1, x, grad_eps);
    auto x_grad_diff = calculate_gradient_diff<T>(numerical_x, analytical_x);
    EXPECT_NEAR(x_grad_diff, T(0), pass_eps);

    auto func2 = [x](mlfe::Tensor& w){
        return conv2d(x, w, {1, 1}, {0, 0});
    };
    auto numerical_w = numerical_gradient_v2(func2, w, grad_eps);
    auto w_grad_diff = calculate_gradient_diff<T>(numerical_w, analytical_w);
    EXPECT_NEAR(w_grad_diff, T(0), pass_eps);
}

TEST(operator, conv2d_k3_s1_p1_grad){  
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    constexpr int b = 1;
    constexpr int ci = 1;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int co = 1;
    constexpr int k1 = 3;
    constexpr int k2 = 3;
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);
    auto x = fn::create_variable({b, hi, wi, ci});
    auto w = fn::create_variable({k1, k2, ci, co});
    auto analytical_x = fn::create_variable(x.shape());
    auto analytical_w = fn::create_variable(w.shape());

    std::generate(x.begin<T>(), x.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(w.begin<T>(), w.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto y = conv2d(x, w, {1, 1}, {1, 1});
    y.backprop();
    std::copy(x.grad().cbegin<T>(), x.grad().cend<T>(),
        analytical_x.begin<T>());
    std::copy(w.grad().cbegin<T>(), w.grad().cend<T>(),
        analytical_w.begin<T>());
    auto func1 = [w](mlfe::Tensor& x){
        return conv2d(x, w, {1, 1}, {1, 1});
    };
    auto numerical_x = numerical_gradient_v2(func1, x, grad_eps);
    auto x_grad_diff = calculate_gradient_diff<T>(numerical_x, analytical_x);
    EXPECT_NEAR(x_grad_diff, T(0), pass_eps);

    auto func2 = [x](mlfe::Tensor& w){
        return conv2d(x, w, {1, 1}, {1, 1});
    };
    auto numerical_w = numerical_gradient_v2(func2, w, grad_eps);
    auto w_grad_diff = calculate_gradient_diff<T>(numerical_w, analytical_w);
    EXPECT_NEAR(w_grad_diff, T(0), pass_eps);
}

TEST(operator, conv2d_k3_s2_p1_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    constexpr int b = 2;
    constexpr int ci = 3;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int co = 1;
    constexpr int k1 = 3;
    constexpr int k2 = 3;
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);
    auto x = fn::create_variable({b, hi, wi, ci});
    auto w = fn::create_variable({k1, k2, ci, co});
    auto analytical_x = fn::create_variable(x.shape());
    auto analytical_w = fn::create_variable(w.shape());

    std::generate(x.begin<T>(), x.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(w.begin<T>(), w.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto y = conv2d(x, w, {1, 1}, {1, 1});
    y.backprop();
    std::copy(x.grad().cbegin<T>(), x.grad().cend<T>(),
        analytical_x.begin<T>());
    std::copy(w.grad().cbegin<T>(), w.grad().cend<T>(),
        analytical_w.begin<T>());
    auto func1 = [w](mlfe::Tensor& x){
        return conv2d(x, w, {1, 1}, {1, 1});
    };
    auto numerical_x = numerical_gradient_v2(func1, x, grad_eps);
    auto x_grad_diff = calculate_gradient_diff<T>(numerical_x, analytical_x);
    EXPECT_NEAR(x_grad_diff, T(0), pass_eps);

    auto func2 = [x](mlfe::Tensor& w){
        return conv2d(x, w, {1, 1}, {1, 1});
    };
    auto numerical_w = numerical_gradient_v2(func2, w, grad_eps);
    auto w_grad_diff = calculate_gradient_diff<T>(numerical_w, analytical_w);
    EXPECT_NEAR(w_grad_diff, T(0), pass_eps);
}
