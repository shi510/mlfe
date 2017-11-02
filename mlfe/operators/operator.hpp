#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__
#include <string>
#include <vector>
#include <memory>
#include "../core/param_def.hpp"
#include "../core/tensor_blob.hpp"

namespace mlfe{

template<class DeviceContext>
class Operator{
public:
    Operator(
             std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> &inputs_,
             std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> &outputs_,
             ParamDef &param_
             ):param(param_){
        for (int i = 0; i < inputs_.size(); ++i) {
            AddInput(inputs_[i]);
        }
        for (int i = 0; i < outputs_.size(); ++i) {
            AddOutput(inputs_[i]);
        }
    }
    
    virtual ~Operator() {}
    
    Operator(const Operator &_op) {
        name = _op.name;
        inputs = _op.inputs;
        outputs = _op.outputs;
        param = _op.param;
    }
    
    std::string Name() {
        return name;
    }
    
    bool SetName(std::string _name) {
        name = _name;
        return true;
    }
    
    bool AddInput(std::shared_ptr<TensorBlob<DeviceContext>> _input) {
        inputs.push_back(_input);
        return true;
    }
    
    bool AddOutput(std::shared_ptr<TensorBlob<DeviceContext>> _output) {
        outputs.push_back(_output);
        return true;
    }
    
    std::shared_ptr<TensorBlob<DeviceContext>> Input(const int _idx) {
        return inputs[_idx];
    }
    
    std::shared_ptr<TensorBlob<DeviceContext>> Output(const int _idx) {
        return outputs[_idx];
    }
    
    int Inputs() {
        return inputs.size();
    }
    
    int Outputs() {
        return outputs.size();
    }
    
    ParamDef & GetParam() {
        return param;
    }
    
    virtual bool Setup() = 0;
    
    virtual std::shared_ptr<TensorBlob<DeviceContext>> Compute(std::shared_ptr<TensorBlob<DeviceContext>> x) = 0;
    
    virtual std::shared_ptr<TensorBlob<DeviceContext>> ComputeGradients(std::shared_ptr<TensorBlob<DeviceContext>> dy) = 0;
    
    virtual void PrintInfo() {}
    
protected:
    Operator() : param("", 0) {}
    
private:
    std::string name;
    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs;
    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs;
    ParamDef param;
};/* class Operater */

} /* namespace mlfe */
#endif /* __OPERATOR_HPP__ */
