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
using VarPair = std::tuple<std::string, Tensor>;
using VecVarPair = std::vector<VarPair>;

public:
    OpDependency(OpDesignContext *odc);
    
    std::string Name() const;

    std::string UniqueName() const;

    VecVarPair Inputs() const;

    VecVarPair Outputs() const;

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

    Builder &Input(VarPair pair);

    Builder &Output(VarPair pair);

    Builder &Attr(Attribution attr);

    OpDependency Finish();

private:
    std::shared_ptr<OpDepData> odd;
};

} // end namespace mlfe
#endif // end ifndef __OP_DEP_HPP__
