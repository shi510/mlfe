#ifndef __GRADIENT_HELPER_HPP__
#define __GRADIENT_HELPER_HPP__
#include "op_design.h"
#include "graph.h"
#include "op_dep.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace mlfe{

class GradientHelper{
public:
    using TensorUmap = std::unordered_map<Tensor, Tensor>;

    GradientHelper(const OpDesignContext *odc);

    virtual TensorUmap compute_gradient(Tensor var,
                                        Tensor dy
                                       ) = 0;

    OpDependency get_opdep() const;

protected:
    OpDependency dep;
    const OpDesignContext *odc;
};

class GradientHelperRegistry{
using HelperPtr = std::shared_ptr<GradientHelper>;
using HelperCreator = std::function<HelperPtr(OpDesignContext *odc)>;
using MapHelper = std::map<std::string, HelperCreator>;
public:
    void Register(std::string name, HelperCreator gh);

    bool Has(const std::string op_name);

    std::vector<std::string> GetAllOpName();

    HelperPtr GetHelper(std::string name, OpDesignContext *odc);

    static GradientHelperRegistry *Get();

private:
    MapHelper registry;
};

struct GradientHelperRegisterer{
    using HelperPtr = std::shared_ptr<GradientHelper>;
    using HelperCreator = std::function<HelperPtr(OpDesignContext *odc)>;

    GradientHelperRegisterer(std::string name, HelperCreator creator);
};

#define REGIST_GRADIENT_HELPER(Name, ...)                      \
    _REGIST_GRADIENT_HELPER_(Name, __VA_ARGS__)

#define _REGIST_GRADIENT_HELPER_(Name, Helper)                 \
static GradientHelperRegisterer                                \
NAME_CONCAT(GradientHelperRegisterer_##Name, __LINE__) =       \
    GradientHelperRegisterer(std::string(#Name),               \
    [](OpDesignContext *odc)->std::shared_ptr<GradientHelper>{ \
        return std::make_shared<Helper>(odc);                  \
    });

} // end namespace mlfe
#endif // end ifndef __GRADIENT_HELPER_HPP__
