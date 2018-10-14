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

    class Builder;

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
public:
    OpDesign GetOpDesign() const;

    Tensor Input(int idx) const;

    Tensor Output(int idx) const;

    template <class AttrType>
    AttrType GetAttr(std::string attr_name) const;

    Attributes AllAttrs() const;

    int NumInput() const;

    int NumOutput() const;

    class Builder;
private:
    OpDesign od;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    Attributes attrs;
};

class OpDesignContext::Builder{
public:
    Builder(OpDesign od);

    Builder &Input(Tensor x);

    Builder &Input(std::vector<Tensor> xs);

    Builder &Output(Tensor y);

    Builder &Output(std::vector<Tensor> ys);

    Builder &Attr(Attributes attrs);

    Builder &Attr(Attribution attr);

    OpDesignContext Finish();

private:
    OpDesignContext odc;
};

template <class AttrType>
AttrType OpDesignContext::GetAttr(std::string attr_name) const{
    if (!attrs.Has(attr_name)){
        throw std::string("OpDesignContext::GetAttr - "
            "No atrribute exists. -> ") + attr_name;
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
