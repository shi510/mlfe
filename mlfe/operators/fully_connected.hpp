#ifndef __FULLY_CONNECTED_OP_HPP__
#define __FULLY_CONNECTED_OP_HPP__

#include "operator.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{
    
template <class DataType, class DeviceContext>
class FullyConnectedOp final : public Operator<DeviceContext>{
public:
    explicit FullyConnectedOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, w, b};
    enum OutputSchema{y};
    TensorBlob<DeviceContext> bias_multiplier;
    int m;
    int n;
    int k;
};
    
template <class DataType, class DeviceContext>
class FullyConnectedGradientOp final : public Operator<DeviceContext>{
public:
    explicit FullyConnectedGradientOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, w, dy};
    enum OutputSchema{dw, db, dx};
    TensorBlob<DeviceContext> bias_multiplier;
    int m;
    int n;
    int k;
};

} /* namespace mlfe */
#endif /* __FULLY_CONNECTED_OP_HPP__ */
