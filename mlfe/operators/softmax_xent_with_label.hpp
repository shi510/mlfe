#ifndef __SOFTMAX_XENT_WITH_LABEL_HPP__
#define __SOFTMAX_XENT_WITH_LABEL_HPP__
#include "operator.hpp"

namespace mlfe{
    
template <class DataType, class DeviceContext>
class SoftmaxCrossEntropyWithLabelOp final : public Operator<DeviceContext>{
public:
    explicit SoftmaxCrossEntropyWithLabelOp(OperatorIO &opio,ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, label};
    enum OutputSchema{prob, loss};
    TensorBlob<DeviceContext> sum_multiplier;
    TensorBlob<DeviceContext> rows_max;
    TensorBlob<DeviceContext> scaler;
    int m;
    int n;
};

template <class DataType, class DeviceContext>
class SoftmaxCrossEntropyWithLabelGradientOp final : public Operator<DeviceContext>{
public:
    explicit SoftmaxCrossEntropyWithLabelGradientOp(OperatorIO &opio,ItemHolder *ih);
    
    void Compute() override;
    
private:
    enum InputSchema{x, label, prob, loss};
    enum OutputSchema{dx};
    int m;
    int n;
};

} /* namespace mlfe */
#endif /* __SOFTMAX_XENT_WITH_LABEL_HPP__ */
