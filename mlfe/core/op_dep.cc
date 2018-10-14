#include "op_dep.h"
#include "attribute.h"
#include "tensor.h"
#include "op_design.h"
#include <unordered_map>
#include <vector>
#include <utility>
#include <sstream>

namespace mlfe{
using OpDep = OpDependency;
using ODB = OpDep::Builder;

struct OpDep::OpDepData{
    std::string op_name;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    Attributes attrs;
    std::shared_ptr<OpDesignContext> odc;
};

OpDep::OpDependency(OpDesignContext *odc){
    odd = std::make_shared<OpDepData>();
}

std::string OpDep::Name() const{
    return odd->op_name;
}

std::string OpDep::UniqueName() const{
    std::stringstream ss;
    ss << "Name:" << odd->op_name;
    for (auto in : odd->inputs) {
        ss << "/In:" << in.Name();
    }
    for (auto out : odd->outputs) {
        ss << "/Out:" << out.Name();
    }
    return ss.str();
}

OpDep::Tensors OpDep::Inputs() const{
    return odd->inputs;
}

OpDep::Tensors OpDep::Outputs() const{
    return odd->outputs;
}

OpDesignContext *OpDep::Context() const{
    return odd->odc.get();
}

ODB::Builder(std::string op_name){
    odd = std::make_shared<OpDepData>();
    odd->op_name = op_name;
}

ODB &ODB::Input(Tensor x){
    odd->inputs.push_back(x);
    return *this;
}

ODB &ODB::Output(Tensor y){
    odd->outputs.push_back(y);
    return *this;
}

ODB &ODB::Attr(Attribution attr){
    odd->attrs.SetAttr(attr);
    return *this;
}

OpDep ODB::Finish(){
    const OpDesignRegistry *reg = OpDesignRegistry::Get();
    const OpDesign od = reg->GetOpDesign(odd->op_name);
    OpDesignContext odc = OpDesignContext::Builder(od)
        .Input(odd->inputs)
        .Output(odd->outputs)
        .Attr(odd->attrs)
        .Finish();

    OpDep dep(&odc);
    odd->odc = std::make_shared<OpDesignContext>(odc);
    *dep.odd = *odd;
    return dep;
}

REGIST_OP(Unknown).Finish();

} // end namespace mlfe;
