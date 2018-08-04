#ifndef __OP_DESIGN_HPP__
#define __OP_DESIGN_HPP__
#include "attribute.h"
#include "tensor.h"
#include <vector>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>

namespace mlfe{
class OpDesign;
class OpDesignContext;

class OpDesign{
using ShapeInferFn = std::function<void(OpDesignContext *)>;
using PairStrStr = std::pair<std::string, std::string>;

public:
    std::string Name() const;

    std::vector<PairStrStr> Input() const;

    std::vector<PairStrStr> Output() const;

    std::vector<PairStrStr> Attr() const;

    ShapeInferFn ShapeInference() const;

    void MakeContext(OpDesignContext *odc) const;

    class Builder;
protected:
    bool VerifyContext(OpDesignContext *odc) const;

    bool VerifyIO(OpDesignContext *odc) const;

    bool VerifyAttrs(OpDesignContext *odc) const;

private:
    std::string name;
    std::vector<PairStrStr> inputs;
    std::vector<PairStrStr> outputs;
    std::vector<PairStrStr> attrs;
    ShapeInferFn shape_fn;
};

class OpDesign::Builder{
public:
    Builder(std::string name);

    Builder &Input(std::string name, std::string type);

    Builder &Output(std::string name, std::string type);

    Builder &Attr(std::string attr_name, std::string attr_type);

    Builder &ShapeInference(ShapeInferFn shape_fn);

    OpDesign Finish() const;

private:
    OpDesign od;
};

class OpDesignContext{
using VarPair = std::tuple<std::string, Tensor>;
public:
    Tensor Input(std::string name) const;

    Tensor Output(std::string name) const;

    template <class AttrType>
    AttrType GetAttr(std::string attr_name) const;

    std::vector<VarPair> AllVars() const;

    Attributes AllAttrs() const;

    class Builder;
private:
    friend class OpDesign;
    std::unordered_map<std::string, Tensor> inputs;
    std::unordered_map<std::string, Tensor> outputs;
    Attributes attrs;
};

class OpDesignContext::Builder{
using VecVarPair = std::vector<VarPair>;
public:
    Builder &Input(VecVarPair xs);

    Builder &Input(VarPair x);

    Builder &Output(VecVarPair ys);

    Builder &Output(VarPair y);

    Builder &Attr(Attributes attrs);

    Builder &Attr(Attribution attr);

    OpDesignContext Finish();

private:
    OpDesignContext odc;
};

template <class AttrType>
AttrType OpDesignContext::GetAttr(std::string attr_name) const{
    if (!attrs.Has(attr_name)){
        throw std::string("OpDesignContext::GetAttr - No atrribute exists. -> ") + attr_name;
    }
    return attrs.GetAttr<AttrType>(attr_name);
}

class OpDesignRegistry{
public:
    void Register(OpDesign oc);

    bool Has(const std::string op_name) const;

    std::vector<std::string> GetAllOpName() const;

    OpDesign GetOpDesign(std::string op_name) const;

    static OpDesignRegistry * Get();

private:
    std::map<std::string, OpDesign> registry;
};

struct OpDesignRegisterer{
    OpDesignRegisterer(OpDesign os);
};

#define REGIST_OP(Name)                            \
static OpDesignRegisterer                          \
NAME_CONCAT(OpDesignRegisterer_##Name, __LINE__) = \
    OpDesign::Builder(std::string(#Name))

#define REGIST_OP_GRAD(Name)                                 \
static OpDesignRegisterer                                    \
NAME_CONCAT(OpDesignRegisterer_##Name##Gradient, __LINE__) = \
    OpDesign::Builder(std::string(#Name)+"Gradient")

} // end namespace mlfe
#endif // end ifndef __OP_DESIGN_HPP__
