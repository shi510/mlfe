#include "tensor.h"
#include "device.h"
#include "attribute.h"
#include "mlfe/operators/basic_arithmetic.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/utils/assert.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>

namespace mlfe{

//TODO : use thread for _children_modified
struct Tensor::impl{
    impl() : __ti(type::float32()), __stop_grad(false){}
    memory_ptr _mem;
    // std::shared_ptr<Tensor> _gradient;
    std::string __name;
    bool __trainable;
    bool __stop_grad;
    type::TypeInfo __ti;
    std::vector<int> __shape;
    int __size;
    std::shared_ptr<class graph> __g;
    node n;
    // node backprop_n;
};

Tensor::Tensor(const bool trainable)
    : _pimpl({.s = std::make_shared<impl>()})
{
    set_trainable(trainable);
    _pimpl->__g = get_default_graph();
}

Tensor::Tensor(std::string name, const bool trainable)
    : _pimpl({.s = std::make_shared<impl>()})
{
    set_trainable(trainable);
    set_name(name);
    _pimpl->__g = get_default_graph();
}

Tensor::Tensor(std::vector<int> shape, const std::string name, const bool trainable)
    : _pimpl({.s = std::make_shared<impl>()})
{
    resize(shape);
    set_trainable(trainable);
    set_name(name);
    _pimpl->__g = get_default_graph();
}

bool Tensor::operator==(const Tensor &v) const{
    return _pimpl.get() == v._pimpl.get();
}

Tensor Tensor::weak_copy(){
    Tensor weak;
    weak._pimpl = {.w = this->_pimpl.s};
    return weak;
}

memory_ptr Tensor::get_memory() const{
    return _pimpl->_mem;
}

std::string Tensor::name() const
{
    return _pimpl->__name;
}

void Tensor::set_name(std::string name)
{
    _pimpl->__name = name;
}

void Tensor::set_trainable(const bool trainable)
{
    _pimpl->__trainable = trainable;
}

void Tensor::stop_gradient(bool stop_grad)
{
    _pimpl->__stop_grad = stop_grad;
}

bool Tensor::trainable() const
{
    return _pimpl->__trainable;
}

void Tensor::reshape(std::vector<int> shape)
{
    auto target_size = std::accumulate(shape.begin(),
        shape.end(), 1, std::multiplies<int>());
    if(target_size != _pimpl->__size)
    {
        std::cerr<<"element size is not match, ";
        std::cerr<<target_size<<" != "<<_pimpl->__size<<std::endl;
        return;
    }
    _pimpl->__shape = shape;
}

void Tensor::resize(std::vector<int> shape, type::TypeInfo ti)
{
    _pimpl->__shape = shape;
    _pimpl->__size = std::accumulate(_pimpl->__shape.begin(),
        _pimpl->__shape.end(), 1, std::multiplies<int>());
    _pimpl->__ti = ti;
    if (_pimpl->_mem) {
        _pimpl->_mem->allocate(this->size() * this->type().size);
    }
    else {
        _pimpl->_mem = create_memory(this->size() * this->type().size);
    }
}

int Tensor::size() const
{
    return _pimpl->__size;
}

int Tensor::dims() const
{
    return _pimpl->__shape.size();
}

int Tensor::dim(int idx) const
{
    int d = 0;
    if(_pimpl->__shape.size() > idx){ d = _pimpl->__shape[idx]; }
    else if(_pimpl->__shape.size() == 0){ d = 0; }
    else { std::runtime_error("Tensor::shape.size() <= idx"); }
    return d;
}

const std::vector<int>& Tensor::shape() const
{
    return _pimpl->__shape;
}

type::TypeInfo Tensor::type() const
{
    return _pimpl->__ti;
}

std::shared_ptr<graph> Tensor::get_graph() const
{
    return _pimpl->__g;
}

void Tensor::set_gradient_v2()
{
    get_node().add_attr("grad_marker", std::vector<std::function<void (Tensor&)>>());
    get_node().add_attr("grad", std::make_shared<Tensor>());
}

void Tensor::set_node(node n)
{
    _pimpl->n = n;
}

node& Tensor::get_node() const
{
    return _pimpl->n;
}

Tensor Tensor::view(std::vector<int> shape){
    Tensor y = *this;
    y.reshape(shape);
    return y;
}

template <>
void Tensor::copy_from(std::vector<float> vec){
    std::copy(vec.begin(), vec.end(), begin<float>());
}

void Tensor::zero(){ operators::set_zeros(*this); }

void Tensor::one(){ operators::set_ones(*this); }

void Tensor::add_grad_marker(std::function<void (Tensor &)> marker)
{
    using T = std::vector<std::function<void (Tensor &)>>;
    (*_pimpl->n.get_attr("grad_marker").data<T>()).push_back(marker);
}

/*
    It back-propagates gradient-information to leaf-nodes.

    TODO: this function should only be enabled on root node.
        1. Check if this node is root.
*/
void Tensor::backprop_v2()
{
    using gm_func_t = std::vector<std::function<void (Tensor &)>>;
    auto topo_list = topological_sort_v2(get_node(), true);
    auto dy = grad_v2();
    //
    // self gradient is 1.
    //
    dy.one();
    for(auto & n : topo_list)
    {
        auto gm_markers = *n.get_attr("grad_marker").data<gm_func_t>();
        dy = **n.get_attr("grad").data<std::shared_ptr<Tensor>>();
        for(auto & gm : gm_markers)
        {
            gm(dy);
        }
    }
}

/*
    Return its gradients.
    backprop_v2 should be called before calling this function.
*/
Tensor Tensor::grad_v2() const
{
    return **_pimpl->n.get_attr("grad").data<std::shared_ptr<Tensor>>();
}

const void *Tensor::_host_data() const {
    return _pimpl->_mem->host_data<void>();
}

void *Tensor::_mutable_host_data(){
    for(auto& o : _pimpl->n.get_outputs())
    {
        o.set_mutation(true);
    }
    return _pimpl->_mem->mutable_host_data<void>();
}

const void *Tensor::_device_data() const {
    return _pimpl->_mem->device_data<void>();
}

void *Tensor::_mutable_device_data(){
    for(auto& o : _pimpl->n.get_outputs())
    {
        o.set_mutation(true);
    }
    return _pimpl->_mem->mutable_device_data<void>();
}

Tensor Tensor::operator+(const Tensor & other) const{
    return operators::add(*this, other);
}

Tensor Tensor::operator-(const Tensor & other) const{
    return operators::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor & other) const{
    return operators::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor & other) const{
    return operators::div(*this, other);
}

/*
 * TODO: Fix computaional graph in +=, -=, *=, /= operators.
 *       Do not use copy, find alternatives.
 *
 */
Tensor & operator+=(Tensor & a, const Tensor & b){
    auto c = operators::add(a, b);
    copy(c.get_memory(), a.get_memory());
    return a;
}

Tensor & operator-=(Tensor & a, const Tensor & b){
    auto c = operators::sub(a, b);
    copy(c.get_memory(), a.get_memory());
    return a;
}

Tensor & operator*=(Tensor & a, const Tensor & b){
    auto c = operators::mul(a, b);
    copy(c.get_memory(), a.get_memory());
    return a;
}

Tensor & operator/=(Tensor & a, const Tensor & b){
    auto c = operators::div(a, b);
    copy(c.get_memory(), a.get_memory());
    return a;
}

namespace functional{

Tensor create_variable(std::vector<int> shape, const bool trainable){
    Tensor var;
    var.set_trainable(trainable);
    var.set_gradient_v2();
    var.resize(shape);
    var.grad_v2().resize(shape);
    var.grad_v2().zero();
    var.get_node().add_attr("tensor", var.weak_copy());
    var.get_node().add_attr("op_name", "variable");
    return var;
}

Tensor create_variable(std::vector<int> shape, type::TypeInfo ti, const bool trainable){
    Tensor var;
    var.set_trainable(trainable);
    var.set_gradient_v2();
    var.resize(shape, ti);
    var.grad_v2().resize(shape);
    return var;
}

} // end namespace functional
} // end namespace mlfe

namespace std{

size_t hash<mlfe::Tensor>::operator()(const mlfe::Tensor &v) const{
    return hash<shared_ptr<mlfe::Tensor::impl>>{}(v._pimpl.shared());
}

} // end namespace std
