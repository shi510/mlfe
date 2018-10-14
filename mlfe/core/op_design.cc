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

OpDesign ODC::GetOpDesign() const{
    return od;
}

Tensor ODC::Input(int idx) const{
    if(inputs.size() < idx){
        throw std::string("OpDesignContext::Input - "
            "too large index");
    }
    return inputs[idx];
}

Tensor ODC::Output(int idx) const{
    if(outputs.size() < idx){
        throw std::string("OpDesignContext::Output - "
            "too large index.");
    }
    return outputs[idx];
}

Attributes ODC::AllAttrs() const{
    return attrs;
}

int ODC::NumInput() const{
    return inputs.size();
}

int ODC::NumOutput() const{
    return outputs.size();
}

using ODCB = ODC::Builder;

ODCB::Builder(OpDesign od){
    odc.od = od;
}

ODCB &ODCB::Input(std::vector<Tensor> xs){
    odc.inputs.insert(odc.inputs.end(), xs.begin(), xs.end());
    return *this;
}

ODCB &ODCB::Input(Tensor x){
    odc.outputs.push_back(x);
    return *this;
}

ODCB &ODCB::Output(std::vector<Tensor> ys){
    odc.outputs.insert(odc.outputs.end(), ys.begin(), ys.end());
    return *this;
}

ODCB &ODCB::Output(Tensor y){
    odc.outputs.push_back(y);
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
    odc.od.ShapeInference()(&odc);
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
