#ifndef __GRADIENT_HELPER_HPP__
#define __GRADIENT_HELPER_HPP__
#include "op_design.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace mlfe{

class GradientHelper{
public:
    using GradientPairs = std::vector<std::pair<Tensor, Tensor>>;
    using HelperOut = std::tuple<Tensor, GradientPairs>;

    GradientHelper(const OpDesignContext *odc);

    virtual HelperOut Get(Tensor dy) = 0;

protected:
    std::string Gradient(std::string var_name);

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
