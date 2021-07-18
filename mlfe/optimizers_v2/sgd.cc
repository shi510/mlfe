#include "mlfe/optimizers_v2/sgd.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace optimizers{
using namespace operators_v2;

SGD::SGD(float lr, float momentum){
    __lr = lr;
    __mm = momentum;
}

void SGD::set_variables(std::vector<Tensor> vars){
    __vars = vars;
    for(int n = 0; n < __vars.size(); ++n){
        auto moment = functional::create_variable(__vars[n].shape());
        std::fill(moment.begin<float>(), moment.end<float>(), 0.f);
        __var_moments.push_back(moment);
    }
}

void SGD::update(){
    for(int n = 0; n < __vars.size(); ++n){
        call<sgd_kernel>(
            marker::I(__vars[n], __vars[n].grad_v2(), __var_moments[n]),
            __lr, __mm, 0.f);
    }
}

} // end namespace optimizers
} // end namespace mlfe
