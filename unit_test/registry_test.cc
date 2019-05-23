#include <gtest/gtest.h>
#include <mlfe/core/op_registry.h>
#include <iostream>
#include <string>

namespace registry_test{
using namespace mlfe;

class test_op : public detail::op_impl<test_op>{
public:
    test_op(){
        this->name = "test_op";
        this->description = "use like this.";
    }

    void run() override;
};

void test_op::run(){
    // TODO:
    //   1. remove using this->is_registered.
    //     ; it is needed to self-register.
    std::cout<<"is registered ? "<<this->is_registered<<std::endl;
}

TEST(registry_test, create){
    auto op = detail::op_registry::create("test_op");
    EXPECT_NE(op, nullptr);
    op->run();
}

} // end namespace registry_test