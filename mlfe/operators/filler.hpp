#ifndef __FILLER_OP_HPP__
#define __FILLER_OP_HPP__
#include <random>
#include "operator.hpp"


namespace mlfe{

template <class DeviceContext>
class FillOp : public Operator<DeviceContext>{
public:
    explicit FillOp(OperatorIO &opio, ItemHolder *ih);
    
protected:
    std::mt19937 rng;
};

template <class DataType, class DeviceContext>
class ConstantFillOp final : public FillOp<DeviceContext>{
public:
    explicit ConstantFillOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum OutputSchema{y};
    DataType val;
};

template <class DataType, class DeviceContext>
class XavierFillOp final : public FillOp<DeviceContext>{
public:
    explicit XavierFillOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum OutputSchema{y};
    std::uniform_real_distribution<DataType> uniform;
    DataType scale;
};

} /* namespace mlfe */
#endif /* __FILLER_OP_HPP__ */
