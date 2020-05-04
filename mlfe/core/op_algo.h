#ifndef __OP_ALGO_HPP__
#define __OP_ALGO_HPP__
#include "mlfe/utils/types.h"
#include "device.h"
#include "tensor.h"
#include "attribute.h"
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <string>

namespace mlfe{

class OpAlgoContext;

class op_algo_runtime_context
{
public:
    void set_training(bool training);

    bool training() const;

private:
    bool __training;
};

class OpAlgo{
public:
    OpAlgo(OpAlgoContext *oac, std::string name = "");

    virtual void Compute(op_algo_runtime_context& rc) = 0;

    virtual void Compute()
    {
        op_algo_runtime_context rc;
        Compute(rc);
    }

    std::string get_name() const{
        return name;
    }
private:
    std::string name;
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
using Tensors = std::vector<Tensor>;
using Workspace = std::map<std::string, memory_ptr>;
public:
    OpAlgoContext(std::string op_name);

    std::string get_op_name() const;

    int num_inputs() const;

    int num_outputs() const;

    Tensor get_input(int idx) const;

    Tensor get_output(int idx) const;

    void add_input(Tensor in);

    void add_output(Tensor out);

    void add_attr(Attribution attr);

    template <class T>
    T get_attr(std::string name) const;

private:
    std::string _op_name;
    Workspace _ws;
    Attributes _attrs;
    Tensors _inputs;
    Tensors _outputs;
};

template <typename T>
T OpAlgoContext::get_attr(std::string name) const{
    return _attrs.GetAttr<T>(name);
}

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
