#ifndef __OP_DEP_HPP__
#define __OP_DEP_HPP__
#include <string>
#include <memory>
#include <vector>
#include <utility>

namespace mlfe{
class OpDesignContext;
class Tensor;
class Attribution;

class OpDependency{
using Tensors = std::vector<Tensor>;

public:
    OpDependency(OpDesignContext *odc);
    
    std::string Name() const;

    std::string UniqueName() const;

    Tensors Inputs() const;

    Tensors Outputs() const;

    OpDesignContext *Context() const;

    OpDependency() = delete;

    class Builder;

private:
    struct OpDepData;
    std::shared_ptr<OpDepData> odd;
};

class OpDependency::Builder{
public:
    Builder(std::string op_name);

    Builder &Input(Tensor x);

    Builder &Output(Tensor y);

    Builder &Attr(Attribution attr);

    OpDependency Finish();

private:
    std::shared_ptr<OpDepData> odd;
};

} // end namespace mlfe
#endif // end ifndef __OP_DEP_HPP__
