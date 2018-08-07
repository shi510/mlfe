#include "op_design.h"
#include <iostream>

namespace mlfe{
using OD = OpDesign;

std::string OD::Name() const{
    return name;
}

std::vector<OD::PairStrStr> OD::Input() const{
    return inputs;
}

std::vector<OD::PairStrStr> OD::Output() const{
    return outputs;
}

std::vector<OD::PairStrStr> OD::Attr() const{
    return attrs;
}

OD::ShapeInferFn OD::ShapeInference() const{
    return shape_fn;
}

void OD::MakeContext(OpDesignContext *odc) const{
    try{
        VerifyContext(odc);
    }
    catch(std::string &e){
        throw e;
    }
    shape_fn(odc);
}

bool OD::VerifyContext(OpDesignContext *odc) const{
    try{
        VerifyIO(odc);
        VerifyAttrs(odc);
    }
    catch(std::string &e){
        throw e;
    }
    return true;
}

bool OD::VerifyIO(OpDesignContext *odc) const{
    const std::string err_msg = "OpDesign::VerifyIO - "
        "Failed to find the variable. -> ";
    for(int n = 0; n < inputs.size(); ++n){
        if(odc->inputs.count(inputs[n].first) <= 0){
            throw err_msg + inputs[n].first;
        }
    }
    for(int n = 0; n < outputs.size(); ++n){
        if(odc->outputs.count(outputs[n].first) <= 0){
            throw err_msg + outputs[n].first;
        }
    }
    return true;
}

bool OD::VerifyAttrs(OpDesignContext *odc) const{
    const std::string err_msg = "OpDesign::VerifyAttrs - "
        "Failed to find the attribute. -> ";
    for(int n = 0; n < attrs.size(); ++n){
        if(!odc->attrs.Has(attrs[n].first)){
            throw err_msg + attrs[n].first;
        }
    }
}

using OB = OD::Builder;

OB::Builder(std::string name){
    od.name = name;
    od.shape_fn = [](OpDesignContext *){};
}

OB &OB::Input(std::string name, std::string type){
    od.inputs.push_back(std::make_pair(name, type));
    return *this;
}

OB &OB::Output(std::string name, std::string type){
    od.outputs.push_back(std::make_pair(name, type));
    return *this;
}

OB &OB::Attr(std::string attr_name, std::string attr_type){
    od.attrs.push_back(std::make_pair(attr_name, attr_type));
    return *this;
}

OB &OB::ShapeInference(OpDesign::ShapeInferFn shape_fn){
    od.shape_fn = shape_fn;
    return *this;
}

OpDesign OB::Finish() const{
    return od;
}

using ODC = OpDesignContext;

Tensor ODC::Input(std::string name) const{
    if(inputs.count(name) <= 0){
        throw std::string("OpDesignContext::Input - "
            "No variable's name. -> ") + name;
    }
    return inputs.find(name)->second;
}

Tensor ODC::Output(std::string name) const{
    if(outputs.count(name) <= 0){
        throw std::string("OpDesignContext::Output - "
            "No variable's name. -> ") + name;
    }
    return outputs.find(name)->second;
}

std::vector<ODC::VarPair> ODC::AllVars() const{
    std::vector<VarPair> vars;
    for(auto var : inputs){
        vars.push_back(var);
    }

    for(auto var : outputs){
        vars.push_back(var);
    }
    return vars;
}

Attributes ODC::AllAttrs() const{
    return attrs;
}

using ODCB = ODC::Builder;

ODCB &ODCB::Input(VecVarPair xs){
    for(int n = 0; n < xs.size(); ++n){
        if(odc.inputs.count(std::get<0>(xs[n])) > 0){
            throw std::string("OpDesignContext::Input - "
                "Variable's name already exists. -> ") + std::get<0>(xs[n]);
        }
        odc.inputs.emplace(std::get<0>(xs[n]), std::get<1>(xs[n]));
    }

    return *this;
}

ODCB &ODCB::Input(ODC::VarPair x){
    if(odc.inputs.count(std::get<0>(x)) > 0){
        throw std::string("OpDesignContext::Input - "
            "Variable's name already exists. -> ") + std::get<0>(x);
    }
    odc.inputs.emplace(std::get<0>(x), std::get<1>(x));

    return *this;
}

ODCB &ODCB::Output(VecVarPair ys){
    for(int n = 0; n < ys.size(); ++n){
        if(odc.outputs.count(std::get<0>(ys[n])) > 0){
            throw std::string("OpDesignContext::Output - "
                "Variable's name already exists. -> ") + std::get<0>(ys[n]);
        }
        odc.outputs.emplace(std::get<0>(ys[n]), std::get<1>(ys[n]));
    }
    return *this;
}

ODCB &ODCB::Output(ODC::VarPair y){
    if(odc.outputs.count(std::get<0>(y)) > 0){
        throw std::string("OpDesignContext::Output - "
            "Variable's name already exists. -> ") + std::get<0>(y);
    }
    odc.outputs.emplace(std::get<0>(y), std::get<1>(y));
    return *this;
}

ODCB &ODCB::Attr(Attributes attrs){
    odc.attrs = attrs;
    return *this;
}

ODCB &ODCB::Attr(Attribution attr){
    odc.attrs.SetAttr(attr);
    return *this;
}

OpDesignContext ODCB::Finish(){
    return odc;
}

void OpDesignRegistry::Register(OpDesign oc){
    if(registry.count(oc.Name()) != 0){
        std::cout << "OpDesignRegistry::Register - "
            "Key already registered. ->" << oc.Name() << std::endl;
        std::exit(1);
    }
    registry[oc.Name()] = oc;
}

bool OpDesignRegistry::Has(const std::string op_name) const{
    return registry.count(op_name) != 0;
}

std::vector<std::string> OpDesignRegistry::GetAllOpName() const{
    std::vector<std::string> op_names;
    auto op_register = OpDesignRegistry::Get();
    for(const auto& pair_iter : op_register->registry){
        op_names.push_back(pair_iter.first);
    }
    return op_names;
}

OpDesign OpDesignRegistry::GetOpDesign(std::string op_name) const{
    return OpDesignRegistry::Get()->registry[op_name];
}

OpDesignRegistry *OpDesignRegistry::Get(){
    static OpDesignRegistry internal_static_register = OpDesignRegistry();
    return &internal_static_register;
}

OpDesignRegisterer::OpDesignRegisterer(OpDesign os){
    OpDesignRegistry::Get()->Register(os);
}
} // end namespace mlfe;
