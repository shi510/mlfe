#include "mlfe/optimizers_v2/adadelta.h"

namespace mlfe{
namespace optimizers{
using namespace operators_v2;

adadelta::adadelta(float lr, float momentum, float eps){
    __lr = lr;
    __momentum = momentum;
    __eps = eps;
}

void adadelta::set_variables(std::vector<Tensor> vars){
    __vars = vars;
    for(int n = 0; n < __vars.size(); ++n){
        auto grad_hist = functional::create_variable(__vars[n].shape());
        auto v_hist = functional::create_variable(__vars[n].shape());
        std::fill(grad_hist.begin<float>(), grad_hist.end<float>(), 0.f);
        std::fill(v_hist.begin<float>(), v_hist.end<float>(), 0.f);
        __grad_hist.push_back(grad_hist);
        __acc_hist.push_back(v_hist);
    }
}

void adadelta::update(){
    for(int n = 0; n < __vars.size(); ++n){
        call<adadelta_kernel>(
            marker::I(__vars[n], __vars[n].grad_v2(), __grad_hist[n], __acc_hist[n]),
            __lr, __momentum, __eps);
    }
}

} // end namespace optimizers
} // end namespace mlfe
