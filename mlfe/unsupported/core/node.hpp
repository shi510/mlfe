#ifndef __NODE_HPP__
#define __NODE_HPP__
#include <memory>
#include "tensor.hpp"
#include "../../core/registry.hpp"
#include "../../core/param_def.hpp"
#include "../core/workspace.hpp"

namespace mlfe { namespace node {

struct OperatorContext {
    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    ParamDef *attr;
};

struct NodeFunctor {
    virtual void Init(OperatorContext *oc) = 0;
    virtual void Run() = 0;
};

class Node{
public:
    Node(std::string name);

    std::string Name();

    void AddParam(std::string name, std::string val);

    std::string GetParam(std::string name);

    void Type(DataType dt);

    void Accel(Accelerator acc);

    DataType Type() const;

    Accelerator Accel() const;

    std::string Input(unsigned int idx) const;

    std::string Output(unsigned int idx) const;

    int Inputs() const;

    int Outputs() const;

    void Init(Workspace *ws);

    void Run();

protected:
    void AddInput(std::string input);

    void AddOutput(std::string input);

    std::string GetInput(unsigned int idx) const;

    std::string GetOutput(unsigned int idx) const;

    unsigned int InputSize() const;

    unsigned int OutputSize() const;

    virtual void InternalInit(Workspace *ws, OperatorContext *oc) = 0;

protected:
    typedef struct InternalNode;
    std::shared_ptr<InternalNode> _inode;
};

class NodeGradient : public Node {
public:
    void Init(Workspace *ws);

    void Run();

protected:
    NodeGradient(std::string name) : Node(name) {}

    virtual void InternalInit(Workspace *ws, OperatorContext *oc) = 0;

    virtual void InternalGradientInit(Workspace *ws, OperatorContext *oc) = 0;
private:
};

template <typename InheritedType>
class NodeIO : public NodeGradient {
public:
    InheritedType &Input(std::string input) {
        AddInput(input);
        return *reinterpret_cast<InheritedType *>(this);
    }

    InheritedType &Output(std::string output) {
        AddOutput(output);
        return *reinterpret_cast<InheritedType *>(this);
    }
    
    template <typename T, 
        typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    InheritedType &Attr(std::string name, T val) {
        AddParam(name, std::to_string(val));
        return *reinterpret_cast<InheritedType *>(this);
    }

    InheritedType &Attr(std::string name, std::string val) {
        AddParam(name, val);
        return *reinterpret_cast<InheritedType *>(this);
    }
    
    std::string Attr(std::string name) {
        return GetParam(name);
    }

protected:
    NodeIO(std::string name) : NodeGradient(name) {}

    virtual void InternalInit(Workspace *ws, OperatorContext *oc) = 0;

    virtual void InternalGradientInit(Workspace *ws, OperatorContext *oc) = 0;
};

std::string CreateNodeName(std::string name, DataType dt, Accelerator accel);

DECLARE_REGISTRY(
    NodeFunctorRegistry,
    std::string,
    std::shared_ptr<NodeFunctor>
)

#define REGIST_NODE_FUNCTOR(Key, Type, Accel, ...)                  \
namespace {   \
    static RegistererNodeFunctorRegistry (NAME_CONCAT(NodeGradientFunctor_##Key, __LINE__))(      \
      CreateNodeName(#Key, Type, Accel),                                                \
      NodeFunctorRegistry(),                                       \
      RegistererNodeFunctorRegistry::DefaultCreator<__VA_ARGS__>   \
    );                                                     \
} // end namespace

DECLARE_REGISTRY(
    NodeGradientFunctorRegistry,
    std::string,
    std::shared_ptr<NodeFunctor>
)

#define REGIST_NODE_GRADIENT_FUNCTOR(Key, Type, Accel, ...)                  \
namespace {   \
    static RegistererNodeGradientFunctorRegistry (NAME_CONCAT(NodeGradientFunctor_##Key, __LINE__))(      \
      CreateNodeName(#Key, Type, Accel),                                               \
      NodeGradientFunctorRegistry(),                                       \
      RegistererNodeGradientFunctorRegistry::DefaultCreator<__VA_ARGS__>   \
    );                                                     \
} // end namespace

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __NODE_HPP__
