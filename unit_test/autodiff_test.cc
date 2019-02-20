#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/operators.h>
#include <algorithm>

namespace autodiff_test{
using namespace mlfe;
namespace fn = functional;

template <typename T>
struct case0{
    case0(){
        one = fn::constant(1, {2, 2});
        three = fn::constant(3, {2, 2});
        plus_1_2 = fn::add(one, fn::constant(2, one.shape()));
        cube_of_3 = plus_1_2 * plus_1_2 * three;
        y = fn::mean(cube_of_3);
        y.eval();
    }

    void backprop(){
        y.backprop();
    }

    Tensor one;
    Tensor plus_1_2;
    Tensor three;
    Tensor cube_of_3;
    Tensor y;
};

} // end namespace autodiff_test

TEST(autodiff_test, case0_eval_check){
    autodiff_test::case0<float> tcase;

    // one = [1, 1, 1, 1]
    std::for_each(tcase.one.data<float>(),
                  tcase.one.data<float>() + tcase.one.size(),
                  [](const float &val){ EXPECT_EQ(val, 1.f); });

    // three = [3, 3, 3, 3]
    std::for_each(tcase.three.data<float>(),
                  tcase.three.data<float>() + tcase.three.size(),
                  [](const float &val){ EXPECT_EQ(val, 3.f); });

    // plus_1_2 = one + 2 = [3, 3, 3, 3]
    std::for_each(tcase.plus_1_2.data<float>(),
                  tcase.plus_1_2.data<float>() + tcase.plus_1_2.size(),
                  [](const float &val){ EXPECT_EQ(val, 3.f); });

    //cube_of_3 = plus_1_2 * plus_1_2 * 3 = [27, 27, 27, 27]
    std::for_each(tcase.cube_of_3.data<float>(),
                  tcase.cube_of_3.data<float>() + tcase.cube_of_3.size(),
                  [](const float &val){ EXPECT_EQ(val, 27.f); });

    //y = sumation(cube_of_3) / len(cube_of_3) = [27]
    std::for_each(tcase.y.data<float>(),
                  tcase.y.data<float>() + tcase.y.size(),
                  [](const float &val){ EXPECT_EQ(val, 27.f); });
}

TEST(autodiff_test, case0_grad_check){
    autodiff_test::case0<float> tcase;
    float answer;

    tcase.backprop();
    // the gradient of root of computation graph is one.
    answer = 1.f;
    std::for_each(tcase.y.grad().data<float>(),
        tcase.y.grad().data<float>() + tcase.y.grad().size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of cube_of_3 = 1 / len(cube_of_3)
    answer = 1.f / tcase.cube_of_3.size();
    std::for_each(tcase.cube_of_3.grad().data<float>(),
        tcase.cube_of_3.grad().data<float>() + tcase.cube_of_3.grad().size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of three = plus_1_2 * plus_1_2 * dy
    answer = 3.f * 3.f * answer;
    std::for_each(tcase.three.grad().data<float>(),
        tcase.three.grad().data<float>() + tcase.three.grad().size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of plus_1_2 = 2 * plus_1_2 * dy
    answer = 2.f * 3.f * 3.f * 1.f / tcase.cube_of_3.size();
    std::for_each(tcase.plus_1_2.grad().data<float>(),
        tcase.plus_1_2.grad().data<float>() + tcase.plus_1_2.grad().size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of one = dy
    std::for_each(tcase.one.grad().data<float>(),
        tcase.one.grad().data<float>() + tcase.one.grad().size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });
}

namespace autodiff_test{
using namespace mlfe;
namespace fn = functional;

template <typename T>
struct case1{
    case1(){
        int init_val;

        x1 = fn::create_variable({2, 2});
        x2 = fn::create_variable({2, 2});
        x3 = fn::create_variable({2, 2});

        sq1 = fn::squared_difference(x1, fn::constant(5, x1.shape()));
        sq2 = fn::squared_difference(x2, fn::constant(3, x2.shape()));
        sq3 = fn::squared_difference(x3, fn::constant(1, x3.shape()));

        mul1 = fn::mul(sq1, fn::constant(-1.5, x1.shape()));
        mul2 = fn::mul(sq2, fn::constant(-1.0, x2.shape()));
        mul3 = fn::mul(sq3, fn::constant(0.5, x3.shape()));

        add_n = fn::add_n({mul1, mul2, mul3});
        mean = fn::mean(add_n);

        init_val = 0;
        std::generate(x1.begin<float>(),
                      x1.end<float>(),
                      [&init_val](){
                          init_val += 1;
                          return init_val;
                      });

        init_val = 0;
        std::generate(x2.begin<float>(),
                      x2.end<float>(),
                      [&init_val](){
                          init_val += 2;
                          return init_val;
                      });

        init_val = 0;
        std::generate(x3.begin<float>(),
                      x3.end<float>(),
                      [&init_val](){
                          init_val -= 3;
                          return init_val;
                      });

        mean.eval();
        mean.backprop();
    }

    Tensor x1, x2, x3;
    Tensor sq1, sq2, sq3;
    Tensor mul1, mul2, mul3;
    Tensor add_n;
    Tensor mean;
};

} // end namespace autodiff_test

TEST(autodiff_test, case1_eval_check){
    using T = float;
    autodiff_test::case1<T> tcase;

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

TEST(autodiff_test, case1_grad_check){
    using T = float;
    autodiff_test::case1<T> tcase;
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

namespace autodiff_test{
using namespace mlfe;
namespace fn = functional;

template <typename T>
struct case2{
    case2(){
        x = fn::create_variable({2, 2});

        sq1 = fn::squared_difference(x, fn::constant(5, x.shape()));
        sq2 = fn::squared_difference(x, fn::constant(3, x.shape()));
        sq3 = fn::squared_difference(x, fn::constant(1, x.shape()));

        mul1 = fn::mul(sq1, fn::constant(-1.5, sq1.shape()));
        mul2 = fn::mul(sq2, fn::constant(-1.0, sq2.shape()));
        mul3 = fn::mul(sq3, fn::constant(0.5, sq3.shape()));

        add_n = fn::add_n({mul1, mul2, mul3});
        mean = fn::mean(add_n);

        x.mutable_data<T>()[0] = 1;
        x.mutable_data<T>()[1] = 3;
        x.mutable_data<T>()[2] = 5;
        x.mutable_data<T>()[3] = 7;

        mean.eval();
        mean.backprop();
    }

    Tensor x;
    Tensor sq1, sq2, sq3;
    Tensor mul1, mul2, mul3;
    Tensor add_n;
    Tensor mean;
};

} // end namespace autodiff_test

TEST(autodiff_test, case2_eval_check){
    using T = float;
    autodiff_test::case2<T> tcase;

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

TEST(autodiff_test, case2_grad_check){
    using T = float;
    autodiff_test::case2<T> tcase;
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

namespace autodiff_test{
using namespace mlfe;
namespace fn = functional;

template <typename T>
struct case3{
    case3(){
        x1 = fn::create_variable({2, 2});
        x2 = fn::create_variable({2, 2});

        sq = fn::squared_difference(x1, x2);

        mul = fn::mul(sq, x1);

        add = fn::add(mul, sq);
        mean = fn::mean(add);

        x1.mutable_data<T>()[0] = 2;
        x1.mutable_data<T>()[1] = 3;
        x1.mutable_data<T>()[2] = 5;
        x1.mutable_data<T>()[3] = 7;

        x2.mutable_data<T>()[0] = 11;
        x2.mutable_data<T>()[1] = 13;
        x2.mutable_data<T>()[2] = 17;
        x2.mutable_data<T>()[3] = 19;

        mean.eval();
        mean.backprop();
    }

    Tensor x1, x2;
    Tensor sq;
    Tensor mul;
    Tensor add;
    Tensor mean;
};

} // end namespace autodiff_test

TEST(autodiff_test, case3_eval_check){
    using T = float;
    autodiff_test::case3<T> tcase;

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

TEST(autodiff_test, case3_grad_check){
    using T = float;
    autodiff_test::case3<T> tcase;

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
    EXPECT_EQ(tcase.sq.grad().data<T>()[1], 1.0);
    EXPECT_EQ(tcase.sq.grad().data<T>()[2], 1.5);
    EXPECT_EQ(tcase.sq.grad().data<T>()[3], 2.0);

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
