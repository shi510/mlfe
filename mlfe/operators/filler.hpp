#ifndef __FILLER_OP_HPP__
#define __FILLER_OP_HPP__

#include "operator.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DeviceContext>
class FillOp : public Operator<DeviceContext>{
public:
    explicit FillOp(
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                    ParamDef param = ParamDef()
                    ) : Operator<DeviceContext>(inputs, outputs, param), rng(math::GetRandomSeed()) {}
    
protected:
    std::mt19937 rng;
};

template <class DataType, class DeviceContext>
    class ConstantFillOp final : public FillOp<DeviceContext>{
public:
    explicit ConstantFillOp(
                            std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                            std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                            ParamDef param = ParamDef()
                            ) : FillOp<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 0, "Input size must be 0.");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        auto y = this->Output(OutputSchema::y);
        if(!param.GetParamByName("Value", val)){
            val = static_cast<DataType>(0);
        }
        if(y->IsEmpty()){
            throw std::string("[ConstantFill] Output is empty.");
        }
        
    }
    
    void Compute() override {
        auto y = this->Output(OutputSchema::y);
        DataType *ptr = y->template GetPtrMutable<DataType>();
        for(int n = 0; n < y->Size(); ++n){
            ptr[n] = val;
        }
    }
    
private:
    enum OutputSchema{y};
    DataType val;
};

template <class DataType, class DeviceContext>
class XavierFillOp final : public FillOp<DeviceContext>{
public:
    explicit XavierFillOp(
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                    ParamDef param = ParamDef()
                    ) : FillOp<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 0, "Input size must be 0.");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        auto y = this->Output(OutputSchema::y);
        if(y->IsEmpty()){
            throw std::string("[XaiverFill] Output is empty.");
        }
        scale = std::sqrt(static_cast<DataType>(2) / static_cast<DataType>(y->Size() / y->Dim(0)));
        uniform = std::uniform_real_distribution<DataType>(-scale, scale);
    }
    
    void Compute() override {
        auto y = this->Output(OutputSchema::y);
        DataType *ptr = y->template GetPtrMutable<DataType>();
        for(int n = 0; n < y->Size(); ++n){
            ptr[n] = uniform(this->rng);
        }
    }
    
private:
    enum OutputSchema{y};
    std::uniform_real_distribution<DataType> uniform;
    DataType scale;
};

} /* namespace mlfe */
#endif /* __FILLER_OP_HPP__ */
