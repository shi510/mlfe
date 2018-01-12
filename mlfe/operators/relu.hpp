#ifndef __RELU_OP_HPP__
#define __RELU_OP_HPP__

#include "operator.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class ReluOp final : public Operator<DeviceContext>{
public:
    explicit ReluOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    bool inplace;
};

template <class DataType, class DeviceContext>
class ReluGradientOp final : public Operator<DeviceContext>{
public:
    explicit ReluGradientOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, dy};
    enum OutputSchema{dx};
    bool inplace;
};

} /* namespace mlfe */
#endif /* __RELU_OP_HPP__ */
