#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/operators.h>

namespace autodiff_test{
using namespace mlfe;
namespace fn = functional;

template <typename T>
struct case0{
    case0(){
        one = fn::constant(1, {2, 2});
        three = fn::constant(3, {2, 2});
        plus_1_2 = fn::add(one, fn::constant(2, one.Shape()));
        cube_of_3 = plus_1_2 * plus_1_2 * three;
        y = fn::mean(cube_of_3);
        y.eval();
    }
    
    void backprob(){
        y.backprob();
    }

    Tensor one;
    Tensor plus_1_2;
    Tensor three;
    Tensor cube_of_3;
    Tensor y;
};

} // end namespace tensor_test

TEST(autodiff_test, eval_check){
    autodiff_test::case0<float> tcase;

    // one = [1, 1, 1, 1]
    std::for_each(tcase.one.data<float>(),
                  tcase.one.data<float>() + tcase.one.Size(),
                  [](const float &val){ EXPECT_EQ(val, 1.f); });

    // three = [3, 3, 3, 3]
    std::for_each(tcase.three.data<float>(),
                  tcase.three.data<float>() + tcase.three.Size(),
                  [](const float &val){ EXPECT_EQ(val, 3.f); });

    // plus_1_2 = one + 2 = [3, 3, 3, 3]
    std::for_each(tcase.plus_1_2.data<float>(),
                  tcase.plus_1_2.data<float>() + tcase.plus_1_2.Size(),
                  [](const float &val){ EXPECT_EQ(val, 3.f); });

    //cube_of_3 = plus_1_2 * plus_1_2 * 3 = [27, 27, 27, 27]
    std::for_each(tcase.cube_of_3.data<float>(),
                  tcase.cube_of_3.data<float>() + tcase.cube_of_3.Size(),
                  [](const float &val){ EXPECT_EQ(val, 27.f); });

    //y = sumation(cube_of_3) / len(cube_of_3) = [27]
    std::for_each(tcase.y.data<float>(),
                  tcase.y.data<float>() + tcase.y.Size(),
                  [](const float &val){ EXPECT_EQ(val, 27.f); });
}

TEST(autodiff_test, grad_check){
    autodiff_test::case0<float> tcase;
    float answer;

    tcase.backprob();
    // the gradient of root of computation graph is one.
    answer = 1.f;
    std::for_each(tcase.y.grad().data<float>(),
        tcase.y.grad().data<float>() + tcase.y.grad().Size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of cube_of_3 = 1 / len(cube_of_3)
    answer = 1.f / tcase.cube_of_3.Size();
    std::for_each(tcase.cube_of_3.grad().data<float>(),
        tcase.cube_of_3.grad().data<float>() + tcase.cube_of_3.grad().Size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of three = plus_1_2 * plus_1_2 * dy
    answer = 3.f * 3.f * answer;
    std::for_each(tcase.three.grad().data<float>(),
        tcase.three.grad().data<float>() + tcase.three.grad().Size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of plus_1_2 = 2 * plus_1_2 * dy
    answer = 2.f * 3.f * 3.f * 1.f / tcase.cube_of_3.Size();
    std::for_each(tcase.plus_1_2.grad().data<float>(),
        tcase.plus_1_2.grad().data<float>() + tcase.plus_1_2.grad().Size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });

    // gradient of one = dy
    answer = answer;
    std::for_each(tcase.one.grad().data<float>(),
        tcase.one.grad().data<float>() + tcase.one.grad().Size(),
        [&](const float &val){ EXPECT_EQ(val, answer); });
}