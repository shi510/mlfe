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
    std::vector<OpDep::VarPair> inputs;
    std::vector<OpDep::VarPair> outputs;
    Attributes attrs;
    std::shared_ptr<OpDesignContext> odc;
};

OpDep::OpDependency(OpDesignContext *odc){
    odd = std::make_shared<OpDepData>();
}

std::string OpDep::Name() const{
    return odd->op_name;
}

std::string OpDep::UniqueName() const {
    std::stringstream ss;
    ss << "Name:" << odd->op_name;
    for (auto in : odd->inputs) {
        ss << "/In:" << std::get<1>(in).Name();
    }
    for (auto out : odd->outputs) {
        ss << "/Out:" << std::get<1>(out).Name();
    }
    return ss.str();
}

OpDep::VecVarPair OpDep::Inputs() const{
    return odd->inputs;
}

OpDep::VecVarPair OpDep::Outputs() const{
    return odd->outputs;
}

OpDesignContext *OpDep::Context() const{
    return odd->odc.get();
}

ODB::Builder(std::string op_name){
    odd = std::make_shared<OpDepData>();
    odd->op_name = op_name;
}

ODB &ODB::Input(OpDep::VarPair pair){
    odd->inputs.push_back(pair);
    return *this;
}

ODB &ODB::Output(OpDep::VarPair pair){
    odd->outputs.push_back(pair);
    return *this;
}

ODB &ODB::Attr(Attribution attr){
    odd->attrs.SetAttr(attr);
    return *this;
}

OpDep ODB::Finish(){
    const OpDesignRegistry *reg = OpDesignRegistry::Get();
    const OpDesign od = reg->GetOpDesign(odd->op_name);
    OpDesignContext odc = OpDesignContext::Builder()
        .Input(odd->inputs)
        .Output(odd->outputs)
        .Attr(odd->attrs)
        .Finish();

    // tensor rehsape inference.
    od.ShapeInference()(&odc);

    OpDep dep(&odc);
    odd->odc = std::make_shared<OpDesignContext>(odc);
    *dep.odd = *odd;

    return dep;
}

REGIST_OP(Unknown).Finish();

} // end namespace mlfe;
