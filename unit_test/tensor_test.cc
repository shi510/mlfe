#include <gtest/gtest.h>
#include <mlfe/core/tensor.h>

namespace tensor_test{
using namespace mlfe;
namespace fn = functional;

template <typename T>
struct case0{
    case0(){
        var = fn::create_variable({1, 3, 5, 7});
    }
    
    void fill_with_direct_access(T val){
        std::fill(var.mutable_data<T>(),
            var.mutable_data<T>() + var.Size(),
            val);
    }

    void fill_with_iterator(T val){
        std::fill(var.begin<T>(), var.end<T>(), val);
    }

    Tensor var;
};

} // end namespace tensor_test

TEST(tensor_test, memory_allocation){
    tensor_test::case0<float> tcase;
    EXPECT_EQ(tcase.var.Size(), 1 * 3 * 5 * 7);
    EXPECT_EQ(tcase.var.Shape()[0], 1);
    EXPECT_EQ(tcase.var.Shape()[1], 3);
    EXPECT_EQ(tcase.var.Shape()[2], 5);
    EXPECT_EQ(tcase.var.Shape()[3], 7);
    EXPECT_NE(tcase.var.data<float>(), nullptr);
}

TEST(tensor_test, memory_constant_fill){
    tensor_test::case0<float> tcase;

    tcase.fill_with_direct_access(135.79f);
    std::for_each(tcase.var.data<float>(),
                  tcase.var.data<float>() + tcase.var.Size(),
                  [](const float &val){ EXPECT_EQ(val, 135.79f); });

    tcase.fill_with_iterator(975.31f);
    std::for_each(tcase.var.begin<float>(),tcase.var.end<float>(),
                  [](const float &val){ EXPECT_EQ(val, 975.31f); });
}