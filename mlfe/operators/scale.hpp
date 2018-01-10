#ifndef __SCALE_OP_HPP__
#define __SCALE_OP_HPP__

#include "operator.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class ScaleOp final : public Operator<DeviceContext>{
public:
    explicit ScaleOp(
                     std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                     std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                     ParamDef param
                     ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 1, "Input size must be 1(x).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        
        if(this->GetParam().GetParamByName("Scale", scaler) &&
           y->IsEmpty() &&
           !x->IsEmpty()
           ){
            y->template Resize<DataType>(x);
        }
        else{
            runtime_assert(x->Dims() == y->Dims(), "x's dim size must be same with y's dim.");
        }
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        math::scal<DataType, DeviceContext>(
                                            x->Size(),
                                            scaler,
                                            x->template GetPtrConst<DataType>(),
                                            y->template GetPtrMutable<DataType>()
                                            );
    }
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    DataType scaler;
};

} /* namespace mlfe */
#endif /* __SCALE_OP_HPP__ */
