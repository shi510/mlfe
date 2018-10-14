#include "tensor.h"
#include "../utils/assert.h"
#include "op_dep.h"
#include "op_design.h"
#include <sstream>

namespace mlfe{

struct Tensor::InternalData{
    InternalData()
        : init_dep(OpDependency::Builder("Unknown").Finish()),
        op_dep(OpDependency::Builder("Unknown").Finish()),
        _exec_order(0),
        _trainable(false){}
    OpDependency op_dep;
    OpDependency init_dep;
    std::vector<Tensor> _parents;
    std::vector<Tensor> _children;
    bool _trainable;
    int _exec_order;
};

using TDA = Tensor::DependencyAdder;

struct TDA::pimpl{
    pimpl();
    OpDependency target_dep;
};

TDA::pimpl::pimpl() :target_dep(OpDependency::Builder("Unknown").Finish()){}

TDA::DependencyAdder(OpDependency dep){
    _impl = std::make_shared<pimpl>();
    _impl->target_dep = dep;
}

Tensor &Tensor::operator=(DependencyAdder dep_adder){
    internal_data->op_dep = dep_adder._impl->target_dep;
    return *this;
}

Tensor::Tensor()
    : internal_data(std::make_shared<InternalData>()){
}

Tensor::Tensor(std::string name)
    : Variable(name),
    internal_data(std::make_shared<InternalData>()){
}

Tensor::Tensor(std::vector<int> shape)
    : Variable(shape),
    internal_data(std::make_shared<InternalData>()){
}

Tensor::~Tensor(){}

std::string Variable::Name() const{
    return *_name + "_" + std::to_string(_id->Id());
}

void Tensor::add_parent(Tensor p){
    internal_data->_parents.push_back(p);
    //p.add_child(*this);
    p.internal_data->_children.push_back(*this);
}

void Tensor::add_child(Tensor c){
    internal_data->_children.push_back(c);
    //c.add_parent(*this);
    c.internal_data->_parents.push_back(*this);
    if(c.internal_data->_exec_order > internal_data->_exec_order){
        internal_data->_exec_order = c.internal_data->_exec_order + 1;
    }
}

std::vector<Tensor> Tensor::get_parents() const{
    return internal_data->_parents;
}

std::vector<Tensor> Tensor::get_children() const{
    return internal_data->_children;
}

int Tensor::get_exec_order() const{
    return internal_data->_exec_order;
}

OpDependency Tensor::get_dep() const{
    return internal_data->op_dep;
}

void Tensor::set_trainable(bool trainable){
    internal_data->_trainable = trainable;
}

bool Tensor::get_trainable() const{
    return internal_data->_trainable;
}

std::string Tensor::info() const{
    std::stringstream ss;
    ss << "node{" << std::endl;
    ss << "    " << "name:" << Name() << std::endl;
    ss << "    " << "op:" << internal_data->op_dep.Name() << std::endl;
    for(auto &c : internal_data->_children){
        ss << "    " << "in:" << c.Name();
        ss << "(op:" << c.internal_data->op_dep.Name() << ")" << std::endl;
    }
    auto shape = Shape();
    ss << "    " << "shape:";
    for(auto &val : shape){
        ss << val << " ";
    }
    ss << std::endl;
    ss << "}";
    return ss.str();
}

bool Tensor::operator==(const Tensor &v) const{
    return internal_data.get() == v.internal_data.get();
}

// TODO : This code is a temporary solution for
//         initialization of tensor values.
//        I'll update this code to make awesome.
void Tensor::Initialize(Tensor init){
    *this = init;
    internal_data->init_dep = internal_data->op_dep;
    internal_data->op_dep = OpDependency::Builder("Unknown").Finish();
}

OpDependency Tensor::InitDependency() const{
    return internal_data->init_dep;
}

unsigned int Variable::UniqueID::_next_gen = 0;

} // end namespace mlfe

namespace std{

size_t hash<mlfe::Tensor>::operator()(const mlfe::Tensor &v) const{
    return hash<shared_ptr<mlfe::Tensor::InternalData>>{}(v.internal_data);
}

} // end namespace std
