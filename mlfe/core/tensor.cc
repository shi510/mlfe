#include "tensor.h"
#include "device.h"
#include "op_algo.h"
#include "attribute.h"
#include "graph.h"
#include "gradient_helper.h"
#include "../operators/initializer.h"
#include "../operators/basic_arithmetics.h"
#include "../utils/assert.h"
#include <algorithm>
#include <sstream>

namespace mlfe{

struct Tensor::InternalData{
    InternalData() : _exec_order(0), _ctx("unknown"){}
    std::vector<Tensor> _parents;
    std::vector<Tensor> _children;
    int _exec_order;
    memory_ptr _mem;
    OpAlgoContext _ctx;
    std::shared_ptr<OpAlgo> _algo;
    std::shared_ptr<Tensor> _gradient;
    Attributes _attrs;
    std::vector<Tensor> _compute_list;
    std::vector<std::shared_ptr<Tensor>> _backward_list;
};

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

bool Tensor::operator==(const Tensor &v) const{
    return internal_data.get() == v.internal_data.get();
}

void Tensor::eval(){
    //compute all children.
    for(auto t : internal_data->_compute_list){
        if(t.internal_data->_algo != nullptr){
            t.internal_data->_algo->Compute();
        }
    }
}

unsigned int Variable::UniqueID::_next_gen = 0;

void Tensor::add_attr(Attribution attr){
    internal_data->_attrs.SetAttr(attr);
}

template <>
bool Tensor::get_attr<bool>(std::string name){
    return internal_data->_attrs.GetAttr<bool>(name);
}

template <>
int Tensor::get_attr<int>(std::string name){
    return internal_data->_attrs.GetAttr<int>(name);
}

template <>
double Tensor::get_attr<double>(std::string name){
    return internal_data->_attrs.GetAttr<double>(name);
}

template <>
float Tensor::get_attr<float>(std::string name){
    return internal_data->_attrs.GetAttr<float>(name);
}

template <>
std::vector<int> Tensor::get_attr<std::vector<int>>(std::string name){
    return internal_data->_attrs.GetAttr<std::vector<int>>(name);
}

OpAlgoContext Tensor::get_context() const{
    return internal_data->_ctx;
}

memory_ptr Tensor::get_memory() const{
    return internal_data->_mem;
}

void Tensor::backprob(){
    if(internal_data->_backward_list.empty()){
        compute_gradient(*this);
    }
    for(auto &var : internal_data->_backward_list){
        var->internal_data->_algo->Compute();
    }
}

Tensor Tensor::grad(){
    return *internal_data->_gradient;
}

const void *Tensor::_host_data(){
    return internal_data->_mem->host_data<void>();
}

void *Tensor::_mutable_host_data(){
    return internal_data->_mem->mutable_host_data<void>();
}

const void *Tensor::_device_data(){
    return internal_data->_mem->device_data<void>();
}

void *Tensor::_mutable_device_data(){
    return internal_data->_mem->mutable_device_data<void>();
}

void Tensor::compute_gradient(const Tensor root){
    using TensorUmap = std::unordered_map<Tensor, std::vector<Tensor>>;
    auto make_ptr = [](Tensor &t){
        return std::make_shared<Tensor>(t);
    };
    TensorUmap dy_collector;

    // top-down seuqnce
    auto v_list = visit_bfs(root);
    // sort by execution order.
    std::sort(v_list.begin(), v_list.end(), [](Tensor v1, Tensor v2){
        return v1.get_exec_order() > v2.get_exec_order();
    });
    // root gradient is 1.
    dy_collector[root].push_back(functional::constant(1, root.Shape()));
    // set root gradient.
    root.internal_data->_gradient = make_ptr(dy_collector[root][0]);
    root.internal_data->_gradient->eval();
    //run computing all gradients.
    for(auto &var : v_list){
        if(var.internal_data->_algo != nullptr){
            auto op_name = var.internal_data->_algo->get_name();
            auto helper = GradientHelperRegistry::Get();
            auto op_grad = helper->GetHelper(op_name, nullptr);
            //add all partial gradients and propagate down.
            auto dy = functional::add_n(dy_collector[var]);
            //calculate input's gradients.
            auto input_grad = op_grad->compute_gradient(var, dy);
            // store all input's gradients.
            for(auto &it : input_grad){
                dy_collector[it.first].push_back(it.second);
                it.first.internal_data->_gradient = make_ptr(it.second);
                root.internal_data->_backward_list.push_back(it.first.internal_data->_gradient);
            }
            //set current variable's gradient, if a partial gradient exists.
            if(dy_collector[var].size() > 1){
                var.internal_data->_gradient = make_ptr(dy);
                root.internal_data->_backward_list.push_back(var.internal_data->_gradient);
            }
        }
    }
}

Tensor::AssignOpFunctor::AssignOpFunctor(Tensor t, OpAlgoContext ctx){
    auto reg = OpAlgoRegistry::Get();
    auto dev = get_enabled_device();
    std::string op_name = ctx.get_op_name();
    std::string full_op_name = "Name:" + op_name + "/Device:";
    std::string dev_name = dev->get_device_name();
    std::string with_accel = dev_name + "(" + dev->get_accelerator_name() + ")";
    
    t.internal_data->_compute_list = visit_bfs(t);
    std::reverse(t.internal_data->_compute_list.begin(), 
                 t.internal_data->_compute_list.end()
                );

    ctx.add_output(t);
    t.internal_data->_ctx = ctx;
    if(reg->Has(full_op_name + with_accel)){
        t.internal_data->_algo = reg->GetOpAlgo(full_op_name + with_accel, &ctx);
    }
    else if(reg->Has(full_op_name + dev_name)){
        t.internal_data->_algo = reg->GetOpAlgo(full_op_name + dev_name, &ctx);
    }
    else{
        throw std::string(op_name) + " is not supported.";
    }
}


namespace functional{

Tensor create_variable(std::vector<int> shape){
    Tensor var;
    var.Reshape(shape);
    var.internal_data->_mem = create_memory(var.Size() * var.Type().size);
    return var;
}

Tensor reshape(Tensor x, std::vector<int> shape){
    Tensor y;
    OpAlgoContext ctx("Reshape");
    y.add_child(x);
    y.internal_data->_algo = nullptr;
    y.internal_data->_gradient = nullptr;
    y.internal_data->_ctx = x.internal_data->_ctx;
    y.internal_data->_mem = x.internal_data->_mem;
    y.Reshape(shape);
    Tensor::AssignOpFunctor(y, ctx);
    return y;
}

void fill(Tensor to, const Tensor from){
    copy(from.get_memory(), to.get_memory());
}

} // end namespace functional
} // end namespace mlfe

namespace std{

size_t hash<mlfe::Tensor>::operator()(const mlfe::Tensor &v) const{
    return hash<shared_ptr<mlfe::Tensor::InternalData>>{}(v.internal_data);
}

} // end namespace std