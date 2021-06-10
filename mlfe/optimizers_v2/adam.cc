#include "mlfe/optimizers_v2/adam.h"
#include "mlfe/math/optimizers.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace optimizers{
using namespace operators_v2;

adam::adam(float lr, float beta1, float beta2, float eps){
    __lr = lr;
    __beta1 = beta1;
    __beta2 = beta2;
    __eps = eps;
}

void adam::set_variables(std::vector<Tensor> vars){
    __vars = vars;
    for(int n = 0; n < __vars.size(); ++n){
        auto m_hist = functional::create_variable(__vars[n].shape());
        auto v_hist = functional::create_variable(__vars[n].shape());
        std::fill(m_hist.begin<float>(), m_hist.end<float>(), 0.f);
        std::fill(v_hist.begin<float>(), v_hist.end<float>(), 0.f);
        __m_hist.push_back(m_hist);
        __v_hist.push_back(v_hist);
    }
}

void adam::update(){
    for(int n = 0; n < __vars.size(); ++n){
        call<adam_kernel>(
            marker::I(__vars[n], __vars[n].grad_v2(), __m_hist[n], __v_hist[n]),
            __lr, __beta1, __beta2, __eps);
    }
}

} // end namespace optimizers
} // end namespace mlfe
