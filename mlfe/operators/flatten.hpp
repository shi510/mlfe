#ifndef __FLATTEN_OP_HPP__
#define __FLATTEN_OP_HPP__

#include "operator.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class FlattenOp final : public Operator<DeviceContext>{
public:
    explicit FlattenOp(
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                    ParamDef param
                    ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 1, "Input size must be 1(x).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        
        if(this->GetParam().GetParamByName("Axis", axis) &&
           y->IsEmpty() &&
           !x->IsEmpty()
           ){
            int flat_from = 1;
            int flat_to = 1;
            *y = x;
            for(int n = 0; n < axis; ++n){
                flat_from *= x->Dim(n);
            }
            for(int n = x->Dims() - 1; n >= axis; --n){
                flat_to *= x->Dim(n);
            }
            y->Reshape({flat_from, flat_to});
        }
        else{
            runtime_assert(x->Dim(0) == y->Dim(0), "x's dim(0) must be same with y's dim(0).");
        }
    }
    
    void Compute() override { }
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    int axis;
};

template <class DataType, class DeviceContext>
class FlattenGradientOp final : public Operator<DeviceContext>{
public:
    explicit FlattenGradientOp(
                            std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                            std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                            ParamDef param = ParamDef()
                            ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 2, "Input size must be 2(x, dy).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(dx).");
        
        const auto x = this->Input(InputSchema::x);
        const auto dy = this->Input(InputSchema::dy);
        auto dx = this->Output(OutputSchema::dx);
        if(dx->IsEmpty() &&
           !x->IsEmpty()
           ){
            *dx = dy;
            dx->Reshape(x);
        }
        else{
            runtime_assert(dx->CompareSizeWith(x) , "dx's size must be same with x.");
        }
    }
    
    void Compute() override { }
    
private:
    enum InputSchema{x, dy};
    enum OutputSchema{dx};
};

} /* namespace mlfe */
#endif /* __FLATTEN_OP_HPP__ */
