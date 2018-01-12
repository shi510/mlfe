#ifndef __FLATTEN_OP_HPP__
#define __FLATTEN_OP_HPP__
#include "operator.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class FlattenOp final : public Operator<DeviceContext>{
public:
    explicit FlattenOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    int axis;
};

template <class DataType, class DeviceContext>
class FlattenGradientOp final : public Operator<DeviceContext>{
public:
    explicit FlattenGradientOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, dy};
    enum OutputSchema{dx};
};

} /* namespace mlfe */
#endif /* __FLATTEN_OP_HPP__ */
