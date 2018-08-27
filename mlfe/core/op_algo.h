#ifndef __OP_ALGO_HPP__
#define __OP_ALGO_HPP__
#include "tensor_mem_ref.h"
#include "../utils/types.h"
#include "op_design.h"
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <string>

namespace mlfe{

class OpAlgoContext;

class OpAlgo{
public:
    OpAlgo(OpAlgoContext *oac);

    virtual void Compute() = 0;
private:
};

class OpAlgoSchema{
using OpAlgoPtr = std::shared_ptr<OpAlgo>;
using OpAlgoCreator = std::function<OpAlgoPtr(OpAlgoContext *)>;
public:
    std::string Name() const;

    std::string Input(std::string name) const;

    std::string Output(std::string name) const;

    std::string Device() const;

    OpAlgoCreator Creator() const;

    class Builder;
private:
    std::string name;
    std::string device;
    std::unordered_map<std::string, std::string> inputs;
    std::unordered_map<std::string, std::string> outputs;
    OpAlgoCreator creator;
};

class OpAlgoSchema::Builder{
public:
    Builder(std::string name);

    Builder &Input(std::string name, std::string type);

    Builder &Output(std::string name, std::string type);

    Builder &Device(std::string device);

    Builder &CreatorFn(OpAlgoCreator fn);

    OpAlgoSchema Finish();
private:
    OpAlgoSchema  oas;
};

class OpAlgoContext{
using VarPtr = std::shared_ptr<TensorMemRef>;
using MapVarPtr = std::map<std::string, Tensor>;
using Workspace = std::map<std::string, VarPtr>;
public:
    OpAlgoContext(Device dev, Workspace *ws, OpDesignContext *odc);

    TensorMemRef *GetVar(std::string name);

    Device GetDevice();

    template <class T>
    T GetAttr(std::string name);

private:
    Device device;
    MapVarPtr vars;
    Workspace *ws;
    Attributes attrs;
};

class OpAlgoRegistry{
using OpAlgoPtr = std::shared_ptr<OpAlgo>;
using MapOpAlgo = std::map<std::string, OpAlgoSchema>;
public:
    void Register(std::string name, OpAlgoSchema oac);

    bool Has(const std::string op_name) const;

    std::vector<std::string> GetAllOpName() const;

    OpAlgoPtr GetOpAlgo(std::string op_name, OpAlgoContext *oac) const;

    static OpAlgoRegistry *Get();

private:
    MapOpAlgo registry;
};

struct OpAlgoRegisterer{
    OpAlgoRegisterer(OpAlgoSchema oas);
};

template <class T>
T OpAlgoContext::GetAttr(std::string name){
    return attrs.GetAttr<T>(name);
}

#define REGIST_OP_ALGO(Name, ...)                           \
    _REGIST_OP_ALGO_(Name, __VA_ARGS__)

#define _REGIST_OP_ALGO_(Name, OpAlgoType)                  \
static OpAlgoRegisterer                                     \
NAME_CONCAT(OpAlgoRegisterer_##Name, __LINE__) =            \
    OpAlgoSchema::Builder(#Name)

#define REGIST_OP_GRAD_ALGO(Name, ...)                      \
    _REGIST_OP_GRAD_ALGO_(Name, __VA_ARGS__)

#define _REGIST_OP_GRAD_ALGO_(Name, OpAlgoType)             \
static OpAlgoRegisterer                                     \
NAME_CONCAT(OpAlgoRegisterer_##Name##Gradient, __LINE__) =  \
    OpAlgoSchema::Builder(std::string(#Name) + "Gradient")

} // end namespace mlfe
#endif // end ifndef __OP_ALGO_HPP__
