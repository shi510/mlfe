#include <gtest/gtest.h>
#include <mlfe/core/tensor.h>
#include <mlfe/operators/basic_arithmetic.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/operators/squared_difference.h>
#include <algorithm>

using namespace mlfe;
namespace fn = functional;
namespace op = operators;

//           mean(y)
//            |
//    -----> mul(x4)
//    |       |
//    |      mul <-----
//    |       |       |
//    |      add(x3) --
//    |     /   \
//   x2    x1    2
template <typename T>
struct case0{
    case0(){
        x1 = Tensor::from_vector<T>({1, 1, 1, 1}, {2, 2});
        x2 = Tensor::from_vector<T>({3, 3, 3, 3}, {2, 2});
        x3 = x1 + 2.f;
        x4 = x3 * x3 * x2;
        y = op::reduce_mean(x4);
    }

    void backprop(){
        y.backprop();
    }

    Tensor x1;
    Tensor x3;
    Tensor x2;
    Tensor x4;
    Tensor y;
};


TEST(autograd, case0_eval_check){
    using T = float;
    case0<T> tcase;

    // x1 = [1, 1, 1, 1]
    std::for_each(tcase.x1.cbegin<T>(), tcase.x1.cend<T>(),
                  [](const T &val){ EXPECT_EQ(val, 1.f); });

    // x2 = [3, 3, 3, 3]
    std::for_each(tcase.x2.cbegin<T>(), tcase.x2.cend<T>(),
                  [](const T &val){ EXPECT_EQ(val, 3.f); });

    // x3 = x1 + 2 = [3, 3, 3, 3]
    std::for_each(tcase.x3.cbegin<T>(), tcase.x3.cend<T>(),
                  [](const T &val){ EXPECT_EQ(val, 3.f); });

    //x4 = x3 * x3 * 3 = [27, 27, 27, 27]
    std::for_each(tcase.x4.cbegin<T>(), tcase.x4.cend<T>(),
                  [](const T &val){ EXPECT_EQ(val, 27.f); });

    //y = sumation(x4) / len(x4) = [27]
    std::for_each(tcase.y.cbegin<T>(), tcase.y.cend<T>(),
                  [](const T &val){ EXPECT_EQ(val, 27.f); });
}

TEST(autograd, case0_grad_check){
    using T = float;
    case0<T> tcase;
    T answer;
    tcase.backprop();
    
    // the gradient of root of computation graph is one.
    answer = 1.f;
    std::for_each(tcase.y.grad().cbegin<T>(),
        tcase.y.grad().cend<T>(),
        [&](const T &val){ EXPECT_EQ(val, answer); });

    // gradient of x4 = 1 / len(x4)
    answer = 1.f / tcase.x4.size();
    std::for_each(tcase.x4.grad().cbegin<T>(),
        tcase.x4.grad().cend<T>(),
        [&](const T &val){ EXPECT_EQ(val, answer); });

    // gradient of x2 = x3 * x3 * dy
    answer = 3.f * 3.f * answer;
    std::for_each(tcase.x2.grad().cbegin<T>(),
        tcase.x2.grad().cend<T>(),
        [&](const T &val){ EXPECT_EQ(val, answer); });

    // gradient of x3 = 2 * x3 * x2 * dy
    answer = 2.f * 3.f * 3.f * 1.f / tcase.x4.size();
    std::for_each(tcase.x3.grad().cbegin<T>(),
        tcase.x3.grad().cend<T>(),
        [&](const T &val){ EXPECT_EQ(val, answer); });

    // gradient of x1 = dy
    std::for_each(tcase.x1.grad().cbegin<T>(),
        tcase.x1.grad().cend<T>(),
        [&](const T &val){ EXPECT_EQ(val, answer); });
}


template <typename T>
struct case1{
    case1(){
        x1 = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
        x2 = Tensor::from_vector<T>({2, 4, 6, 8}, {2, 2});
        x3 = Tensor::from_vector<T>({-3, -6, -9, -12}, {2, 2});

        sq1 = op::squared_difference(x1, Tensor::from_scalar<T>(5, x1.shape()));
        sq2 = op::squared_difference(x2, Tensor::from_scalar<T>(3, x2.shape()));
        sq3 = op::squared_difference(x3, Tensor::from_scalar<T>(1, x3.shape()));

        mul1 = -1.5f * sq1;
        mul2 = -1.0f * sq2;
        mul3 = 0.5f * sq3;

        add_n = mul1 + mul2 + mul3;//op::add_n({mul1, mul2, mul3});
        mean = op::reduce_mean(add_n);
    }

    void backprop(){
        mean.backprop();
    }

    Tensor x1, x2, x3;
    Tensor sq1, sq2, sq3;
    Tensor mul1, mul2, mul3;
    Tensor add_n;
    Tensor mean;
};


TEST(autograd, case1_eval_check){
    using T = float;
    case1<T> tcase;

    EXPECT_EQ(tcase.sq1.data<T>()[0], 16);
    EXPECT_EQ(tcase.sq1.data<T>()[1], 9);
    EXPECT_EQ(tcase.sq1.data<T>()[2], 4);
    EXPECT_EQ(tcase.sq1.data<T>()[3], 1);

    EXPECT_EQ(tcase.sq2.data<T>()[0], 1);
    EXPECT_EQ(tcase.sq2.data<T>()[1], 1);
    EXPECT_EQ(tcase.sq2.data<T>()[2], 9);
    EXPECT_EQ(tcase.sq2.data<T>()[3], 25);

    EXPECT_EQ(tcase.sq3.data<T>()[0], 16);
    EXPECT_EQ(tcase.sq3.data<T>()[1], 49);
    EXPECT_EQ(tcase.sq3.data<T>()[2], 100);
    EXPECT_EQ(tcase.sq3.data<T>()[3], 169);

    EXPECT_EQ(tcase.mul1.data<T>()[0], -24);
    EXPECT_EQ(tcase.mul1.data<T>()[1], -13.5);
    EXPECT_EQ(tcase.mul1.data<T>()[2], -6);
    EXPECT_EQ(tcase.mul1.data<T>()[3], -1.5);

    EXPECT_EQ(tcase.mul2.data<T>()[0], -1);
    EXPECT_EQ(tcase.mul2.data<T>()[1], -1);
    EXPECT_EQ(tcase.mul2.data<T>()[2], -9);
    EXPECT_EQ(tcase.mul2.data<T>()[3], -25);

    EXPECT_EQ(tcase.mul3.data<T>()[0], 8);
    EXPECT_EQ(tcase.mul3.data<T>()[1], 24.5);
    EXPECT_EQ(tcase.mul3.data<T>()[2], 50);
    EXPECT_EQ(tcase.mul3.data<T>()[3], 84.5);

    EXPECT_EQ(tcase.add_n.data<T>()[0], -17);
    EXPECT_EQ(tcase.add_n.data<T>()[1], 10);
    EXPECT_EQ(tcase.add_n.data<T>()[2], 35);
    EXPECT_EQ(tcase.add_n.data<T>()[3], 58);

    EXPECT_EQ(tcase.mean.data<T>()[0], 21.5);
}

TEST(autograd, case1_grad_check){
    using T = float;
    case1<T> tcase;
    tcase.backprop();
    
    EXPECT_EQ(tcase.mean.grad().data<T>()[0], 1);

    EXPECT_EQ(tcase.add_n.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.add_n.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.add_n.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.add_n.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul3.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul3.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul3.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul3.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul2.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul2.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul2.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul2.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul1.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul1.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul1.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul1.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.sq3.grad().data<T>()[0], 0.125);
    EXPECT_EQ(tcase.sq3.grad().data<T>()[1], 0.125);
    EXPECT_EQ(tcase.sq3.grad().data<T>()[2], 0.125);
    EXPECT_EQ(tcase.sq3.grad().data<T>()[3], 0.125);

    EXPECT_EQ(tcase.sq2.grad().data<T>()[0], -0.25);
    EXPECT_EQ(tcase.sq2.grad().data<T>()[1], -0.25);
    EXPECT_EQ(tcase.sq2.grad().data<T>()[2], -0.25);
    EXPECT_EQ(tcase.sq2.grad().data<T>()[3], -0.25);

    EXPECT_EQ(tcase.sq1.grad().data<T>()[0], -0.375);
    EXPECT_EQ(tcase.sq1.grad().data<T>()[1], -0.375);
    EXPECT_EQ(tcase.sq1.grad().data<T>()[2], -0.375);
    EXPECT_EQ(tcase.sq1.grad().data<T>()[3], -0.375);

    EXPECT_EQ(tcase.x3.grad().data<T>()[0], -1);
    EXPECT_EQ(tcase.x3.grad().data<T>()[1], -1.75);
    EXPECT_EQ(tcase.x3.grad().data<T>()[2], -2.5);
    EXPECT_EQ(tcase.x3.grad().data<T>()[3], -3.25);

    EXPECT_EQ(tcase.x2.grad().data<T>()[0], 0.5);
    EXPECT_EQ(tcase.x2.grad().data<T>()[1], -0.5);
    EXPECT_EQ(tcase.x2.grad().data<T>()[2], -1.5);
    EXPECT_EQ(tcase.x2.grad().data<T>()[3], -2.5);

    EXPECT_EQ(tcase.x1.grad().data<T>()[0], 3);
    EXPECT_EQ(tcase.x1.grad().data<T>()[1], 2.25);
    EXPECT_EQ(tcase.x1.grad().data<T>()[2], 1.5);
    EXPECT_EQ(tcase.x1.grad().data<T>()[3], 0.75);
}


//                   mean
//                    |
//                  add_n
//                    |
//           /-----------------\
//          /         |         \
//        mul        mul        mul
//        |  \        | \        | \
//        |   -1.5    |  -1.0    |  0.5
//        |           |          |
//     sq_diff     sq_diff     sq_diff
//     /  |           |  \       |  \
//    5    \          |   3     /    1
//          \---------x--------/
template <typename T>
struct case2{
    case2(){
        x = Tensor::from_vector<T>({1, 3, 5, 7}, {2, 2});

        sq1 = op::squared_difference(x, Tensor::from_scalar<T>(5, x.shape()));
        sq2 = op::squared_difference(x, Tensor::from_scalar<T>(3, x.shape()));
        sq3 = op::squared_difference(x, Tensor::from_scalar<T>(1, x.shape()));

        mul1 = op::mul(sq1, Tensor::from_scalar<T>(-1.5, sq1.shape()));
        mul2 = op::mul(sq2, Tensor::from_scalar<T>(-1.0, sq2.shape()));
        mul3 = op::mul(sq3, Tensor::from_scalar<T>(0.5, sq3.shape()));

        add_n = mul1 + mul2 + mul3;
        mean = op::reduce_mean(add_n);
    }

    void backprop(){
        mean.backprop();
    }

    Tensor x;
    Tensor sq1, sq2, sq3;
    Tensor mul1, mul2, mul3;
    Tensor add_n;
    Tensor mean;
};

TEST(autograd, case2_eval_check){
    using T = float;
    case2<T> tcase;

    EXPECT_EQ(tcase.sq1.data<T>()[0], 16);
    EXPECT_EQ(tcase.sq1.data<T>()[1], 4);
    EXPECT_EQ(tcase.sq1.data<T>()[2], 0);
    EXPECT_EQ(tcase.sq1.data<T>()[3], 4);

    EXPECT_EQ(tcase.sq2.data<T>()[0], 4);
    EXPECT_EQ(tcase.sq2.data<T>()[1], 0);
    EXPECT_EQ(tcase.sq2.data<T>()[2], 4);
    EXPECT_EQ(tcase.sq2.data<T>()[3], 16);

    EXPECT_EQ(tcase.sq3.data<T>()[0], 0);
    EXPECT_EQ(tcase.sq3.data<T>()[1], 4);
    EXPECT_EQ(tcase.sq3.data<T>()[2], 16);
    EXPECT_EQ(tcase.sq3.data<T>()[3], 36);

    EXPECT_EQ(tcase.mul1.data<T>()[0], -24);
    EXPECT_EQ(tcase.mul1.data<T>()[1], -6);
    EXPECT_EQ(tcase.mul1.data<T>()[2], 0);
    EXPECT_EQ(tcase.mul1.data<T>()[3], -6);

    EXPECT_EQ(tcase.mul2.data<T>()[0], -4);
    EXPECT_EQ(tcase.mul2.data<T>()[1], 0);
    EXPECT_EQ(tcase.mul2.data<T>()[2], -4);
    EXPECT_EQ(tcase.mul2.data<T>()[3], -16);

    EXPECT_EQ(tcase.mul3.data<T>()[0], 0);
    EXPECT_EQ(tcase.mul3.data<T>()[1], 2);
    EXPECT_EQ(tcase.mul3.data<T>()[2], 8);
    EXPECT_EQ(tcase.mul3.data<T>()[3], 18);

    EXPECT_EQ(tcase.add_n.data<T>()[0], -28);
    EXPECT_EQ(tcase.add_n.data<T>()[1], -4);
    EXPECT_EQ(tcase.add_n.data<T>()[2], 4);
    EXPECT_EQ(tcase.add_n.data<T>()[3], -4);

    EXPECT_EQ(tcase.mean.data<T>()[0], -8);
}

TEST(autograd, case2_grad_check){
    using T = float;
    case2<T> tcase;
    tcase.backprop();
    EXPECT_EQ(tcase.mean.grad().data<T>()[0], 1);

    EXPECT_EQ(tcase.add_n.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.add_n.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.add_n.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.add_n.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul3.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul3.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul3.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul3.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul2.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul2.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul2.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul2.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul1.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul1.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul1.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul1.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.sq3.grad().data<T>()[0], 0.125);
    EXPECT_EQ(tcase.sq3.grad().data<T>()[1], 0.125);
    EXPECT_EQ(tcase.sq3.grad().data<T>()[2], 0.125);
    EXPECT_EQ(tcase.sq3.grad().data<T>()[3], 0.125);

    EXPECT_EQ(tcase.sq2.grad().data<T>()[0], -0.25);
    EXPECT_EQ(tcase.sq2.grad().data<T>()[1], -0.25);
    EXPECT_EQ(tcase.sq2.grad().data<T>()[2], -0.25);
    EXPECT_EQ(tcase.sq2.grad().data<T>()[3], -0.25);

    EXPECT_EQ(tcase.sq1.grad().data<T>()[0], -0.375);
    EXPECT_EQ(tcase.sq1.grad().data<T>()[1], -0.375);
    EXPECT_EQ(tcase.sq1.grad().data<T>()[2], -0.375);
    EXPECT_EQ(tcase.sq1.grad().data<T>()[3], -0.375);

    EXPECT_EQ(tcase.x.grad().data<T>()[0], 4);
    EXPECT_EQ(tcase.x.grad().data<T>()[1], 2);
    EXPECT_EQ(tcase.x.grad().data<T>()[2], 0);
    EXPECT_EQ(tcase.x.grad().data<T>()[3], -2);
}

//              mean
//               |
//              add <---
//               |     |
//         ---> mul    |
//         |     |     |
//         |   sq_diff -
//         |   /    \
//         - x1      x2
template <typename T>
struct case3{
    case3(){
        x1 = Tensor::from_vector<T>({2, 3, 5, 7}, {2, 2});
        x2 = Tensor::from_vector<T>({11, 13, 17, 19}, {2, 2});
        sq = op::squared_difference(x1, x2);
        mul = op::mul(sq, x1);
        add = op::add(mul, sq);
        mean = op::reduce_mean(add);
    }

    void backprop(){
        mean.backprop();
    }

    Tensor x1, x2;
    Tensor sq;
    Tensor mul;
    Tensor add;
    Tensor mean;
};

TEST(autograd, case3_eval_check){
    using T = float;
    case3<T> tcase;

    EXPECT_EQ(tcase.sq.data<T>()[0], 81);
    EXPECT_EQ(tcase.sq.data<T>()[1], 100);
    EXPECT_EQ(tcase.sq.data<T>()[2], 144);
    EXPECT_EQ(tcase.sq.data<T>()[3], 144);

    EXPECT_EQ(tcase.mul.data<T>()[0], 162);
    EXPECT_EQ(tcase.mul.data<T>()[1], 300);
    EXPECT_EQ(tcase.mul.data<T>()[2], 720);
    EXPECT_EQ(tcase.mul.data<T>()[3], 1008);

    EXPECT_EQ(tcase.add.data<T>()[0], 243);
    EXPECT_EQ(tcase.add.data<T>()[1], 400);
    EXPECT_EQ(tcase.add.data<T>()[2], 864);
    EXPECT_EQ(tcase.add.data<T>()[3], 1152);

    EXPECT_EQ(tcase.mean.data<T>()[0], 664.75);
}

TEST(autograd, case3_grad_check){
    using T = float;
    case3<T> tcase;
    tcase.backprop();

    EXPECT_EQ(tcase.mean.grad().data<T>()[0], 1);

    EXPECT_EQ(tcase.add.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.add.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.add.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.add.grad().data<T>()[3], 0.25);

    EXPECT_EQ(tcase.mul.grad().data<T>()[0], 0.25);
    EXPECT_EQ(tcase.mul.grad().data<T>()[1], 0.25);
    EXPECT_EQ(tcase.mul.grad().data<T>()[2], 0.25);
    EXPECT_EQ(tcase.mul.grad().data<T>()[3], 0.25);

    // d_add/d_sq + (d_add/d_mul)*(d_mul/d_sq) = 0.25 + 0.25 * x1
    EXPECT_EQ(tcase.sq.grad().data<T>()[0], 0.75);
    EXPECT_EQ(tcase.sq.grad().data<T>()[1], 1);
    EXPECT_EQ(tcase.sq.grad().data<T>()[2], 1.5);
    EXPECT_EQ(tcase.sq.grad().data<T>()[3], 2);

    // sq.grad() * d_sq/d_x2
    EXPECT_EQ(tcase.x2.grad().data<T>()[0], 13.5);
    EXPECT_EQ(tcase.x2.grad().data<T>()[1], 20);
    EXPECT_EQ(tcase.x2.grad().data<T>()[2], 36);
    EXPECT_EQ(tcase.x2.grad().data<T>()[3], 48);

    // sq.grad() * d_sq/d_x1 + mul.grad() * sq
    EXPECT_EQ(tcase.x1.grad().data<T>()[0], 6.75);
    EXPECT_EQ(tcase.x1.grad().data<T>()[1], 5);
    EXPECT_EQ(tcase.x1.grad().data<T>()[2], 0);
    EXPECT_EQ(tcase.x1.grad().data<T>()[3], -12);
}
