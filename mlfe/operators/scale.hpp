#ifndef __SCALE_OP_HPP__
#define __SCALE_OP_HPP__

#include "operator.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class ScaleOp final : public Operator<DeviceContext>{
public:
    explicit ScaleOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    DataType scaler;
};

} /* namespace mlfe */
#endif /* __SCALE_OP_HPP__ */
