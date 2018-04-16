#ifndef __FLATTEN_OP_HPP__
#define __FLATTEN_OP_HPP__
#include "operator.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class FlattenOp final : public Operator<DeviceContext>{
public:
    explicit FlattenOp(
        OperatorIO &opio, 
        ItemHolder *ih
    ) : Operator<DeviceContext>(opio, ih) {
        runtime_assert(this->inputs.size() == 1,
            "[Flatten Op] inputs.size() == 1.");
        runtime_assert(this->outputs.size() == 1,
            "[Flatten Op] outputs.size() == 1.");
        const auto x = this->inputs[InputSchema::x];
        auto y = this->outputs[OutputSchema::y];

        runtime_assert(opio.param.HasParam("Axis"), "[Flatten Op] Not found Axis param.");
        if (y->IsEmpty() &&
            !x->IsEmpty()
            ) {
            int flat_from = 1;
            int flat_to = 1;
            axis = opio.param.GetParam<int>("Axis");
            *y = *x;
            for (int n = 0; n < axis; ++n) {
                flat_from *= x->Dim(n);
            }
            for (int n = x->Dims() - 1; n >= axis; --n) {
                flat_to *= x->Dim(n);
            }
            y->Reshape({ flat_from, flat_to });
        }
        else {
            runtime_assert(x->Dim(0) == y->Dim(0),
                "[Flatten Op] x->Dim(0) == y->Dim(0).");
        }
    }
    
    void Compute() override {}
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    int axis;
};

template <class DataType, class DeviceContext>
class FlattenGradientOp final : public Operator<DeviceContext>{
public:
    explicit FlattenGradientOp(
        OperatorIO &opio, 
        ItemHolder *ih
    ) : Operator<DeviceContext>(opio, ih) {
        runtime_assert(this->inputs.size() == 2,
            "[Flatten Gradient Op] inputs.size() == 2.");
        runtime_assert(this->outputs.size() == 1,
            "[Flatten Gradient Op] outputs.size() == 1.");

        const auto x = this->inputs[InputSchema::x];
        const auto dy = this->inputs[InputSchema::dy];
        auto dx = this->outputs[OutputSchema::dx];
        if (dx->IsEmpty() &&
            !x->IsEmpty()
            ) {
            *dx = *dy;
            dx->Reshape(*x);
        }
        else {
            runtime_assert(dx->CompareSizeWith(*x),
                "[Flatten Gradient Op] dx->CompareSizeWith(x).");
        }
    }
    
    void Compute() override {}
    
private:
    enum InputSchema{x, dy};
    enum OutputSchema{dx};
};

} /* namespace mlfe */
#endif /* __FLATTEN_OP_HPP__ */
