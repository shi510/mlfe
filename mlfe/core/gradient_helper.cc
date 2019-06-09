#include "gradient_helper.h"
#include "mlfe/core/tensor.h"
#include <numeric>
#include <iostream>

namespace mlfe{
using GH = GradientHelper;

GH::GradientHelper(){}

using GHR = GradientHelperRegistry;

void GHR::Register(std::string name, HelperCreator creator){
    if(registry.count(name) != 0){
        std::cout << "GradientHelperRegistry.Register : \
            Key already registered. ->" << name << std::endl;
        std::exit(1);
    }
    registry[name] = creator;
}

bool GHR::Has(const std::string op_name){
    return registry.count(op_name) != 0;
}

std::vector<std::string> GHR::GetAllOpName(){
    std::vector<std::string> op_names;
    auto op_register = GHR::Get();
    for(const auto& pair_iter : op_register->registry){
        op_names.push_back(pair_iter.first);
    }
    return op_names;
}

GHR::HelperPtr GHR::GetHelper(std::string name){
    if(registry.count(name) <= 0){
        throw std::string("GradientHelperRegistry.GetHelper : "
            "Not found for ") + name;
    }
    return registry[name]();
}

GHR *GHR::Get(){
    static GradientHelperRegistry internal_static_register = GHR();
    return &internal_static_register;
}

using GHRR = GradientHelperRegisterer;

GHRR::GradientHelperRegisterer(std::string name, HelperCreator creator){
    GHR::Get()->Register(name, creator);
}

class IdentityGrad : public GradientHelper{
public:
    IdentityGrad(){}
    
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        in_grads.push_back(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Identity, IdentityGrad)

} // end namespace mlfe;
