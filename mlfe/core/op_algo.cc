#include "op_algo.h"
#include <iostream>

namespace mlfe{

void op_algo_runtime_context::set_training(bool training)
{
    __training = training;
}

bool op_algo_runtime_context::training() const
{
    return __training;
}

OpAlgo::OpAlgo(OpAlgoContext *oac, std::string name){
    this->name = name;
}

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
    oas.name += "/Device:" + oas.device;
    return oas;
}

OpAlgoContext::OpAlgoContext(std::string op_name){
    _op_name = op_name;
}

std::string OpAlgoContext::get_op_name() const{
    return _op_name;
}

int OpAlgoContext::num_inputs() const{
    return _inputs.size();
}

int OpAlgoContext::num_outputs() const{
    return _outputs.size();
}

Tensor OpAlgoContext::get_input(int idx) const{
    if(_inputs.size() <= idx){
        throw std::string("OpAlgoContext::get_input - index too large.");
    }
    return _inputs[idx];
}

Tensor OpAlgoContext::get_output(int idx) const{
    if(_outputs.size() <= idx){
        throw std::string("OpAlgoContext::get_output - index too large.");
    }
    return _outputs[idx];
}

void OpAlgoContext::add_input(Tensor in){
    _inputs.push_back(in);
}

void OpAlgoContext::add_output(Tensor out){
    _outputs.push_back(out);
}

void OpAlgoContext::add_attr(Attribution attr){
    _attrs.SetAttr(attr);
}

Attributes OpAlgoContext::get_attrs() const
{
    return _attrs;
}

void OpAlgoContext::set_attrs(Attributes attrs)
{
    _attrs = attrs;
}

bool OpAlgoContext::has_attr(std::string name) const
{
    return _attrs.Has(name);
}

using OAR = OpAlgoRegistry;

void OAR::Register(std::string name, OpAlgoSchema oac){
    if(registry.count(name) != 0){
        std::cout << "OpAlgoRegistry::Register - "
            "Key already registered. ->" << name << std::endl;
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
    if(registry.find(op_name) == registry.end()){
        throw std::string("OpAlgoRegistry::GetOpAlgo - "
            "Not found for ") + op_name;
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

std::shared_ptr<OpAlgo> find_op(OpAlgoContext& ctx)
{
    std::shared_ptr<OpAlgo> op = nullptr;
    auto reg = OpAlgoRegistry::Get();
    auto dev = get_enabled_device();
    std::string op_name = ctx.get_op_name();
    std::string full_op_name = "Name:" + op_name + "/Device:";
    std::string dev_name = dev->get_device_name();
    std::string with_accel = dev_name + "(" + dev->get_accelerator_name() + ")";

    if(reg->Has(full_op_name + with_accel)){
        op = reg->GetOpAlgo(full_op_name + with_accel, &ctx);
    }
    else if(reg->Has(full_op_name + dev_name)){
        op = reg->GetOpAlgo(full_op_name + dev_name, &ctx);
    }
    else if(reg->Has(full_op_name + "Any")){
        op = reg->GetOpAlgo(full_op_name + "Any", &ctx);
    }
    return op;
}

template <class Tp>
class Identity : public OpAlgo{
public:
    Identity(OpAlgoContext *oac) : OpAlgo(oac, "Identity"){}
    
    void Compute(op_algo_runtime_context& rc) override{}
};

REGIST_OP_ALGO(Identity)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("Any")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Identity<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class IdentityGrad : public OpAlgo{
public:
    IdentityGrad(OpAlgoContext *oac) : OpAlgo(oac){}
    
    void Compute(op_algo_runtime_context& rc) override{}
};

REGIST_OP_GRAD_ALGO(Identity)
    .Input("X", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("Any")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = IdentityGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe;
