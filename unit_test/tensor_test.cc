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
    using T = float;
    tensor_test::case0<T> tcase;
    EXPECT_EQ(tcase.var.Size(), 1 * 3 * 5 * 7);
    EXPECT_EQ(tcase.var.Shape()[0], 1);
    EXPECT_EQ(tcase.var.Shape()[1], 3);
    EXPECT_EQ(tcase.var.Shape()[2], 5);
    EXPECT_EQ(tcase.var.Shape()[3], 7);
    EXPECT_NE(tcase.var.data<T>(), nullptr);
}

TEST(tensor_test, memory_constant_fill){
    using T = float;
    tensor_test::case0<T> tcase;

    tcase.fill_with_direct_access(135.79f);
    std::for_each(tcase.var.data<T>(),
                  tcase.var.data<T>() + tcase.var.Size(),
                  [](const T &val){ EXPECT_EQ(val, 135.79f); });

    tcase.fill_with_iterator(975.31f);
    std::for_each(tcase.var.begin<T>(),tcase.var.end<T>(),
                  [](const T &val){ EXPECT_EQ(val, 975.31f); });
}

TEST(tensor_test, iterator_increment){
    using T = float;
    tensor_test::case0<T> tcase;
    auto ptr_dirt = tcase.var.data<T>();
    auto ptr_iter = tcase.var.cbegin<T>();
    for(int n = 0; n < tcase.var.Size(); ++n){
        EXPECT_EQ(ptr_dirt, ptr_iter.operator->());
        ptr_dirt++;
        ptr_iter++;
    }
    ptr_dirt = tcase.var.data<T>();
    ptr_iter = tcase.var.cbegin<T>();
    for(int n = 0; n < tcase.var.Size(); ++n){
        EXPECT_EQ(ptr_dirt, ptr_iter.operator->());
        ++ptr_dirt;
        ++ptr_iter;
    }

    auto mptr_dirt = tcase.var.mutable_data<T>();
    auto mptr_iter = tcase.var.begin<T>();
    for(int n = 0; n < tcase.var.Size(); ++n){
        EXPECT_EQ(mptr_dirt, mptr_iter.operator->());
        ++mptr_dirt;
        ++mptr_iter;
    }

    mptr_dirt = tcase.var.mutable_data<T>();
    mptr_iter = tcase.var.begin<T>();
    for(int n = 0; n < tcase.var.Size(); ++n){
        EXPECT_EQ(mptr_dirt, mptr_iter.operator->());
        ++mptr_dirt;
        ++mptr_iter;
    }
}