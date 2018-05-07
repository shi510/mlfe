#include "node.hpp"

namespace mlfe { namespace node {

struct Node::InternalNode {
    std::string name;
    std::vector<std::string> _inputs;
    std::vector<std::string> _outputs;
    ParamDef _pd;
    std::shared_ptr<NodeFunctor> _nf;
    std::shared_ptr<NodeFunctor> _nf_grad;
    DataType _dt;
    Accelerator _acc;
};

Node::Node(std::string name) {
    _inode = std::make_shared<InternalNode>();
    _inode->name = name;
}

void Node::Type(DataType dt) {
    _inode->_dt = dt;
}

void Node::Accel(Accelerator acc) {
    _inode->_acc = acc;
}

void Node::AddParam(std::string name, std::string val) {
    _inode->_pd.Add(name, val);
}

std::string Node::GetParam(std::string name) {
    return _inode->_pd.GetParam<std::string>(name);
}

std::string Node::Name() {
    return _inode->name;
}

DataType Node::Type() const {
    return _inode->_dt;
}

Accelerator Node::Accel() const {
    return _inode->_acc;
}

std::string Node::Input(unsigned int idx) const{
    return GetInput(idx);
}

std::string Node::Output(unsigned int idx) const{
    return GetOutput(idx);
}

int Node::Inputs() const {
    return InputSize();
}

int Node::Outputs() const {
    return OutputSize();
}

void Node::Init(Workspace *ws) {
    std::string type_str = to_string(Type());
    std::string acc_str = to_string(Accel());
    std::string f_name = _inode->name + "_" + type_str + "_" + acc_str;
    OperatorContext oc;
    if (!acc_str.compare(to_string(Accelerator::CUDNN))) {
        if (!NodeFunctorRegistry()->Has(f_name)) {
            acc_str = to_string(Accelerator::CUDA);
        }
    }
    f_name = _inode->name + "_" + type_str + "_" + acc_str;
    if (!NodeFunctorRegistry()->Has(f_name)) {
        throw std::string("Can not find the functor : ") + f_name;
    }
    oc.attr = &_inode->_pd;
    _inode->_nf = NodeFunctorRegistry()->Create(f_name);
    InternalInit(ws, &oc);
    _inode->_nf->Init(&oc);
}

void Node::Run() {
    _inode->_nf->Run();
}

void Node::AddInput(std::string input) {
    _inode->_inputs.push_back(input);
}

void Node::AddOutput(std::string output) {
    _inode->_outputs.push_back(output);
}

std::string Node::GetInput(unsigned int idx) const{
    return _inode->_inputs[idx];
}

std::string Node::GetOutput(unsigned int idx) const{
    return _inode->_outputs[idx];
}

unsigned int Node::InputSize() const {
    return _inode->_inputs.size();
}

unsigned int Node::OutputSize() const {
    return _inode->_outputs.size();
}

ParamDef *Node::GetParam() {
    return &_inode->_pd;
}

void NodeGradient::Init(Workspace *ws) {
    std::string name = _inode->name;
    std::string type_str = to_string(Type());
    std::string acc_str = to_string(Accel());
    std::string grad_name = name + "_" + type_str + "_" + acc_str;
    OperatorContext oc;
    if (!acc_str.compare(to_string(Accelerator::CUDNN))) {
        if (!NodeGradientFunctorRegistry()->Has(grad_name)) {
            acc_str = to_string(Accelerator::CUDA);
        }
    }
    grad_name = name + "_" + type_str + "_" + acc_str;
    if (!NodeGradientFunctorRegistry()->Has(grad_name)) {
        throw std::string("Can not find the functor : ") + grad_name;
    }
    oc.attr = &_inode->_pd;
    _inode->_nf_grad = NodeGradientFunctorRegistry()->Create(grad_name);
    InternalGradientInit(ws, &oc);
    _inode->_nf_grad->Init(&oc);
}

void NodeGradient::Run() {
    _inode->_nf_grad->Run();
}

std::string CreateNodeName(std::string name, DataType dt, Accelerator accel) {
    return name + "_" + to_string(dt) + "_" + to_string(accel);
}

DEFINE_REGISTRY(
    NodeFunctorRegistry,
    std::string,
    std::shared_ptr<NodeFunctor>
)

DEFINE_REGISTRY(
    NodeGradientFunctorRegistry,
    std::string,
    std::shared_ptr<NodeFunctor>
)

} // end namespace node
} // end namespace mlfe