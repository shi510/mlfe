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
    explicit ReluOp(
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                    ParamDef param = ParamDef()
                    ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 1, "Input size must be 1(x).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        inplace = false;
        this->GetParam().GetParamByName("Inplace", inplace);
        
        if(y->IsEmpty() &&
           !x->IsEmpty()
           ){
            if(inplace){
                *y = x;
            }
            else{
                y->template Resize<DataType>(x);
            }
        }
        else{
            runtime_assert(x->Dim(0) == y->Dim(0), "x's dim(0) must be same with y's dim(0).");
        }
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        math::ReluFunction<DataType, DeviceContext>(
                                                    x->Size(),
                                                    x->template GetPtrConst<DataType>(),
                                                    y->template GetPtrMutable<DataType>()
                                                    );
    }
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    bool inplace;
};

template <class DataType, class DeviceContext>
class ReluGradientOp final : public Operator<DeviceContext>{
public:
    explicit ReluGradientOp(
                            std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                            std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                            ParamDef param = ParamDef()
                            ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 2, "Input size must be 2(x, dy).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(dx).");
        
        const auto x = this->Input(InputSchema::x);
        const auto dy = this->Input(InputSchema::dy);
        auto dx = this->Output(OutputSchema::dx);
        inplace = false;
        this->GetParam().GetParamByName("Inplace", inplace);
        
        if(dx->IsEmpty() &&
           !x->IsEmpty()
           ){
            if(inplace){
                *dx = dy;
            }
            else{
                dx->template Resize<DataType>(x);
            }
        }
        else{
            runtime_assert(dx->CompareSizeWith(x) , "dx's size must be same with x.");
        }
        
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        const auto dy = this->Input(InputSchema::dy);
        auto dx = this->Output(OutputSchema::dx);
        math::ReluGradientFunction<DataType, DeviceContext>(
                                                            dy->Size(),
                                                            x->template GetPtrConst<DataType>(),
                                                            dy->template GetPtrConst<DataType>(),
                                                            dx->template GetPtrMutable<DataType>()
                                                            );
    }
    
private:
    enum InputSchema{x, dy};
    enum OutputSchema{dx};
    bool inplace;
};

} /* namespace mlfe */
#endif /* __RELU_OP_HPP__ */
