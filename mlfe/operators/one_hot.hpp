#ifndef __ONE_HOT_OP_HPP__
#define __ONE_HOT_OP_HPP__

#include "operator.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class OneHotOp final : public Operator<DeviceContext>{
public:
    explicit OneHotOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    int dim;
};

} /* namespace mlfe */
#endif /* __ONE_HOT_OP_HPP__ */
