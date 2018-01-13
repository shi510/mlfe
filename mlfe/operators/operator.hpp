#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__
#include <string>
#include <vector>
#include <memory>
#include "../core/param_def.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/registry.hpp"
#include "../core/item_holder.hpp"

namespace mlfe{

struct OperatorIO{
    std::string type;
    std::string name;
    std::string accelerator;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    ParamDef param;
};

class OperatorBase{
public:
    OperatorBase(OperatorIO &opio, ItemHolder *ih);
    
    Item *Inputs(const int idx);
    
    Item *Outputs(const int idx);
    
    int Inputs();
    
    int Outputs();
    
    OperatorIO & GetOperatorIO();
    
    virtual void Compute() = 0;
    
protected:
    OperatorIO opio;
    ItemHolder *ih;
};

template<class DC>
class Operator : public OperatorBase{
public:
    Operator(OperatorIO &opio, ItemHolder *ih)
        : OperatorBase(opio, ih){
            for(auto &in : opio.inputs){
                ih->AddItem<TensorBlob<DC>>(in);
                inputs.push_back(ih->GetItem(in)->Get<TensorBlob<DC>>());
            }
            for(auto &out : opio.outputs){
                ih->AddItem<TensorBlob<DC>>(out);
                outputs.push_back(ih->GetItem(out)->Get<TensorBlob<DC>>());
            }
        }
    
    virtual void Compute() = 0;
    
protected:
    std::vector<TensorBlob<DC> *> inputs;
    std::vector<TensorBlob<DC> *> outputs;
};/* class Operater */
    
struct GradientIO{
    GradientIO(){}
    virtual OperatorIO GetGradientIO(OperatorIO opio) = 0;
};

DECLARE_REGISTRY(
                 OperatorCPU,
                 std::string,
                 std::shared_ptr<OperatorBase>,
                 OperatorIO &,
                 ItemHolder *
                 )
    
DECLARE_REGISTRY(
                 OperatorGradientIO,
                 std::string,
                 std::shared_ptr<GradientIO>
                 )

#define REGIST_OPERATOR_CPU(Key, ...)                  \
namespace {   \
static RegistererOperatorCPU (OperatorCPU_##Key)(      \
  #Key,                                                \
  OperatorCPU(),                                       \
  RegistererOperatorCPU::DefaultCreator<__VA_ARGS__>   \
);                                                     \
}

#define REGIST_OPERATOR_GRADIENT_IO(Key, ...)                  \
namespace {   \
static RegistererOperatorGradientIO (OperatorGradientIO##Key)(      \
#Key,                                                \
OperatorGradientIO(),                                       \
RegistererOperatorGradientIO::DefaultCreator<__VA_ARGS__>   \
);                                                     \
}

std::shared_ptr<OperatorBase>
    CreateOperator(OperatorIO &opio, ItemHolder *ih);
    
std::shared_ptr<OperatorBase>
    CreateOperatorGradient(OperatorIO &opio, ItemHolder *ih);

} /* namespace mlfe */
#endif /* __OPERATOR_HPP__ */
