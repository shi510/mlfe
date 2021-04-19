#include "tensor.h"
#include "device.h"
#include "op_algo.h"
#include "attribute.h"
#include "gradient_helper.h"
#include "mlfe/operators/initializer.h"
#include "mlfe/operators/basic_arithmetics.h"
#include "mlfe/utils/assert.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>

namespace mlfe{

//TODO : use thread for _children_modified
struct Tensor::impl{
    impl() : _ctx("unknown"), __ti(type::float32()), __stop_grad(false){}
    memory_ptr _mem;
    OpAlgoContext _ctx;
    std::shared_ptr<Tensor> _gradient;
    std::string __name;
    bool __trainable;
    bool __stop_grad;
    type::TypeInfo __ti;
    std::vector<int> __shape;
    int __size;
    std::shared_ptr<class graph> __g;
    node n;
    node backprop_n;
};

Tensor::Tensor(const bool trainable)
    : _pimpl(std::make_shared<impl>())
{
    set_trainable(trainable);
    _pimpl->__g = get_default_graph();
}

Tensor::Tensor(std::string name, const bool trainable)
    : _pimpl(std::make_shared<impl>())
{
    set_trainable(trainable);
    set_name(name);
    _pimpl->__g = get_default_graph();
}

Tensor::Tensor(std::vector<int> shape, const std::string name, const bool trainable)
    : _pimpl(std::make_shared<impl>())
{
    resize(shape);
    set_trainable(trainable);
    set_name(name);
    _pimpl->__g = get_default_graph();
}

bool Tensor::operator==(const Tensor &v) const{
    return _pimpl.get() == v._pimpl.get();
}

void Tensor::set_context(OpAlgoContext ctx)
{
    _pimpl->_ctx = ctx;
    auto op = find_op(ctx);
    this->get_node().set_task(make_task([](decltype(op) op, node n) {
        op_algo_runtime_context rc;
        rc.set_training(n.get_graph()->training());
        op->Compute(rc);
        }, op, this->get_node()));
    this->get_backprop_node().set_task(make_task([]() {}));
    for(int n = 0; n < ctx.num_inputs(); ++n)
    {
        this->get_node().add_input(ctx.get_input(n).get_node());
    }
    this->get_node().add_attr("op_name", ctx.get_op_name());
    this->get_node().add_attr("op", op);
    this->get_node().add_attr("tensor", *this);
}

OpAlgoContext & Tensor::get_context() const{
    return _pimpl->_ctx;
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
    return _pimpl->__shape[idx];
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

void Tensor::set_gradient(Tensor t)
{
    _pimpl->_gradient = std::make_shared<Tensor>(t);
}

void Tensor::set_gradient_v2()
{
    get_node().add_attr("grad_marker", std::vector<std::function<void (Tensor)>>());
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

void Tensor::set_backprop_node(node n)
{
    _pimpl->backprop_n = n;
}

node& Tensor::get_backprop_node() const
{
    return _pimpl->backprop_n;
}

void Tensor::eval()
{
    _pimpl->n.run_only_mutated();
}

void Tensor::backprop()
{
    using TensorUmap = std::unordered_map<Tensor, std::vector<Tensor>>;
    TensorUmap dy_collector;
    auto grad_helper = GradientHelperRegistry::Get();
    auto topo_list = topological_sort(_pimpl->n);
    std::sort(topo_list.begin(), topo_list.end(), [](node a, node b) {
        return a.get_name() > b.get_name();
        });
    topo_list.erase(std::unique(topo_list.begin(), topo_list.end()), topo_list.end());
    std::sort(topo_list.begin(), topo_list.end(), [](node a, node b) {
        return a.get_order() > b.get_order();
        });
    auto root_grad = functional::constant(1, this->shape());
    this->set_backprop_node(root_grad.get_node());
    this->set_gradient(root_grad);
    dy_collector[*this].push_back(root_grad);
    for(auto& n : topo_list)
    {
        auto op_name = *n.get_attr("op_name").data<std::string>();
        if (op_name != "None")
        {
            auto op_grad = grad_helper->GetHelper(op_name);
            auto y = *n.get_attr("tensor").data<Tensor>();
            auto dy = functional::add_n(dy_collector[y]);
            if (dy_collector[y].size() > 1)
            {
                y.set_backprop_node(dy.get_node());
                y.set_gradient(dy);
            }
            op_grad->compute_gradient(y, dy);
            for(auto& in : y.get_node().get_inputs())
            {
                auto x = *in.get_attr("tensor").data<Tensor>();
                dy_collector[x].push_back(x.grad());
            }
        }
    }
}

Tensor Tensor::grad(){
    return *_pimpl->_gradient;
}

template <>
void Tensor::copy_from(std::vector<float> vec){
    std::copy(vec.begin(), vec.end(), begin<float>());
}

void Tensor::zero(){
    using T = float;
    std::transform(cbegin<T>(), cend<T>(),
        begin<T>(), [](const T & x){ return T(0);});
}

void Tensor::one(){
    using T = float;
    std::transform(cbegin<T>(), cend<T>(),
        begin<T>(), [](const T & x){ return T(1);});
}

void Tensor::add_grad_marker(std::function<void (Tensor)> marker)
{
    using T = std::vector<std::function<void (Tensor)>>;
    (*_pimpl->n.get_attr("grad_marker").data<T>()).push_back(marker);
}

/*
    It back-propagates gradient-information to leaf-nodes.

    TODO: this function should only be enabled on root node.
        1. Check if this node is root.
*/
void Tensor::backprop_v2() const
{
    using T = std::vector<std::function<void (Tensor)>>;
    auto topo_list = topological_sort_v2(get_node(), true);
    auto dy = grad_v2();
    //
    // self gradient is 1.
    //
    dy.one();
    for(auto n : topo_list)
    {
        T gm_markers = *n.get_attr("grad_marker").data<T>();
        dy = *(*n.get_attr("grad").data<std::shared_ptr<Tensor>>());
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
Tensor & Tensor::grad_v2() const
{
    return **_pimpl->n.get_attr("grad").data<Tensor *>();
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

Tensor & Tensor::operator-=(const Tensor & x){
    using T = float;
    const T * in_ptr = x.data<T>();
    T * out_ptr = mutable_data<T>();
    for(int n = 0; n < x.size(); ++n){
        out_ptr[n] -= in_ptr[n];
    }
    return *this;
}

Tensor Tensor::operator*(const float & val) const{
    using T = float;
    Tensor y = functional::create_variable(this->shape());
    const T * x_ptr = this->data<T>();
    T * y_ptr = y.mutable_data<T>();
    for(int n = 0; n < this->size(); ++n){
        y_ptr[n] = val * x_ptr[n];
    }
    return y;
}

Tensor Tensor::operator-(const Tensor & other) const{
    using T = float;
    Tensor y = functional::create_variable(this->shape());
    const T * x_ptr = this->data<T>();
    const T * other_ptr = other.data<T>();
    T * y_ptr = y.mutable_data<T>();
    for(int n = 0; n < this->size(); ++n){
        y_ptr[n] = x_ptr[n] - other_ptr[n];
    }
    return y;
}

Tensor operator*(const float & val, const Tensor & x){ return x * val; }

namespace functional{

Tensor create_variable(std::vector<int> shape, const bool trainable){
    Tensor var;
    OpAlgoContext ctx("Identity");
    ctx.add_output(var);
    var.set_trainable(trainable);
    var.set_context(ctx);
    var.set_gradient_v2();
    var.resize(shape);
    var.grad_v2().resize(shape);
    return var;
}

Tensor create_variable(std::vector<int> shape, type::TypeInfo ti, const bool trainable){
    Tensor var;
    OpAlgoContext ctx("Identity");
    ctx.add_output(var);
    var.set_trainable(trainable);
    var.set_gradient_v2();
    var.resize(shape, ti);
    var.grad_v2().resize(shape);
    var.set_context(ctx);
    var.get_node().add_attr("grad_marker", std::vector<std::function<void (Tensor)>>());
    var.get_node().add_attr("tensor", var);
    return var;
}

Tensor reshape(Tensor x, std::vector<int> shape){
    OpAlgoContext ctx("Reshape");
    Tensor shape_t = create_variable({ (int)shape.size() }, type::int64());
    Tensor y;
    for(int n = 0; n < shape.size(); ++n){
        shape_t.mutable_data<int64_t>()[n] = shape[n];
    }
    y._pimpl->_gradient = nullptr;
    y._pimpl->_mem = x._pimpl->_mem;
    y._pimpl->__size = x.size();
    y._pimpl->__shape = shape;
    ctx.add_input(x);
    ctx.add_input(shape_t);
    ctx.add_output(y);
    y.set_context(ctx);
    y.get_node().set_task(make_task([]() {}));
    y.get_backprop_node().set_task(make_task([]() {}));
    y.get_node().add_attr("tensor", y);
    return y;
}

} // end namespace functional
} // end namespace mlfe

namespace std{

size_t hash<mlfe::Tensor>::operator()(const mlfe::Tensor &v) const{
    return hash<shared_ptr<mlfe::Tensor::impl>>{}(v._pimpl);
}

} // end namespace std
