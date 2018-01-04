#ifndef __ONE_HOT_OP_HPP__
#define __ONE_HOT_OP_HPP__

#include "operator.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class OneHotOp final : public Operator<DeviceContext>{
public:
    explicit OneHotOp(
                     std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                     std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                     ParamDef param
                     ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 1, "Input size must be 1(x).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        
        if(this->GetParam().GetParamByName("Dim", dim) &&
           y->IsEmpty() &&
           !x->IsEmpty()
           ){
            y->template Resize<DataType>({x->Dim(0), dim});
        }
        else{
            runtime_assert(x->Dims() == y->Dims(), "x's dim size must be same with y's dim.");
        }
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        math::scal<DataType, DeviceContext>(
                                            y->Size(),
                                            static_cast<DataType>(0),
                                            y->template GetPtrConst<DataType>(),
                                            y->template GetPtrMutable<DataType>()
                                            );
        for(int b = 0; b < x->Dim(0); ++b){
            int val = x->template GetPtrConst<DataType>()[b];
            y->template GetPtrMutable<DataType>()[b * dim + val] = static_cast<DataType>(1);
        }
    }
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    int dim;
};

} /* namespace mlfe */
#endif /* __ONE_HOT_OP_HPP__ */
