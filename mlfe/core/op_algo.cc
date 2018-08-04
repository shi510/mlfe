#include "op_algo.h"
#include <iostream>

namespace mlfe{

OpAlgo::OpAlgo(OpAlgoContext *oac){}

using OAS = OpAlgoSchema;

std::string OAS::Name() const{
    return name;
}

std::string OAS::Input(std::string name) const{
    return inputs.find(name)->second;;
}

std::string OAS::Output(std::string name) const{
    return outputs.find(name)->second;
}

std::string OAS::Device() const{
    return device;
}

OAS::OpAlgoCreator OAS::Creator() const{
    return creator;
}

using OASB = OAS::Builder;

OASB::Builder(std::string name){
    oas.name = name;
}

OASB &OASB::Input(std::string name, std::string type){
    oas.inputs[name] = type;
    return *this;
}

OASB &OASB::Output(std::string name, std::string type){
    oas.outputs[name] = type;
    return *this;
}

OASB &OASB::Device(std::string device){
    oas.device = device;
    return *this;
}

OASB &OASB::CreatorFn(OAS::OpAlgoCreator fn){
    oas.creator = fn;
    return *this;
}

OpAlgoSchema OASB::Finish(){
    oas.name = "Name:" + oas.name;
    for(auto in : oas.inputs){
        oas.name += "/In:" + in.second;
    }
    for(auto out : oas.outputs){
        oas.name += "/Out:" + out.second;
    }
    oas.name += "/Device:" + oas.device;
    return oas;
}

OpAlgoContext::OpAlgoContext(Workspace *ws, OpDesignContext *odc){
    this->ws = ws;
    auto vars = odc->AllVars();
    for(auto var_pair : vars){
        this->vars[std::get<0>(var_pair)] = std::get<1>(var_pair);
    }
    attrs = odc->AllAttrs();
}

TensorMemRef *OpAlgoContext::GetVar(std::string name){
    if(vars.count(name) <= 0){
        throw std::string("OpAlgoContext::GetVar - Not found for ") + name;
    }

    return ws->find(vars[name].Name())->second.get();
}

using OAR = OpAlgoRegistry;

void OAR::Register(std::string name, OpAlgoSchema oac){
    if(registry.count(name) != 0){
        std::cout << "OpAlgoRegistry::Register - \
            Key already registered. ->" << name << std::endl;
        std::exit(1);
    }
    registry[name] = oac;
}

bool OAR::Has(const std::string op_name) const{
    return registry.count(op_name) != 0;
}

std::vector<std::string> OAR::GetAllOpName() const{
    std::vector<std::string> op_names;
    auto op_register = OpAlgoRegistry::Get();
    for(const auto& pair_iter : op_register->registry){
        op_names.push_back(pair_iter.first);
    }
    return op_names;
}

OAR::OpAlgoPtr OAR::GetOpAlgo(std::string op_name, OpAlgoContext *oac) const{
    if(registry.count(op_name) <= 0){
        throw std::string("OpAlgoRegistry::GetOpAlgo - \
            Not found for ") + op_name;
    }
    return registry.find(op_name)->second.Creator()(oac);
}

OAR *OAR::Get(){
    static OAR internal_static_register = OAR();
    return &internal_static_register;
}

OpAlgoRegisterer::OpAlgoRegisterer(OpAlgoSchema oas){
    OpAlgoRegistry::Get()->Register(oas.Name(), oas);
}

} // end namespace mlfe;
