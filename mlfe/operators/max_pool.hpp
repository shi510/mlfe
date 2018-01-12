#ifndef __MAX_POOL_OP_HPP__
#define __MAX_POOL_OP_HPP__

#include <cfloat>
#include "operator.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class MaxPoolOp final : public Operator<DeviceContext>{
public:
    explicit MaxPoolOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x};
    enum OutputSchema{y, idx};
    std::vector<int> kernel;
    std::vector<int> stride;
    int out_h, out_w;
};

template <class DataType, class DeviceContext>
class MaxPoolGradientOp final : public Operator<DeviceContext>{
public:
    explicit MaxPoolGradientOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, idx, dy};
    enum OutputSchema{dx};
    std::vector<int> kernel;
    std::vector<int> stride;
    int out_h, out_w;
};

} /* namespace mlfe */
#endif /* __MAX_POOL_OP_HPP__ */
