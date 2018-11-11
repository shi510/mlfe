#include "gradient_descent.h"
#include "../core/tensor.h"
#include "../core/device.h"
#include "../math/optimizers.h"
#include "../math/basic_functions.h"
#if defined(OPTION_USE_CUDNN) || defined(OPTION_USE_CUDA)
#include "../device_context/cuda_context.h"
#else
#include "../device_context/cpu_context.h"
#endif
#include <unordered_map>

namespace mlfe{
namespace opt{

class gradient_descent : public optimizer{
public:
    gradient_descent(double lr, double momentum);

    void apply(Tensor var, Tensor var_grad) override;

private:
    double _lr;
    double _mm;
    std::unordered_map<Tensor, memory_ptr> _var_hist;
};

gradient_descent::gradient_descent(double lr, double momentum)
    : _lr(lr), _mm(momentum){}

void gradient_descent::apply(Tensor var, Tensor var_grad){
#if defined(OPTION_USE_CUDNN) || defined(OPTION_USE_CUDA)
    if(_var_hist.find(var_grad) == _var_hist.end()){
        auto mem = create_memory(var_grad.size() * sizeof(float));
        math::set<float, CUDAContext>(
            var_grad.size(),
            static_cast<float>(0),
            mem->mutable_device_data<float>()
            );
        _var_hist[var_grad] = mem;
    }
    math::gradient_descent_momentum<float, CUDAContext>(
        var.size(),
        var.mutable_device_data<float>(),
        var_grad.mutable_device_data<float>(),
        _var_hist[var_grad]->mutable_device_data<float>(),
        static_cast<float>(_lr),
        static_cast<float>(_mm),
        static_cast<float>(0)
        );
#else
    if(_var_hist.find(var_grad) == _var_hist.end()){
        auto mem = create_memory(var_grad.size() * sizeof(float));
        math::set<float, CPUContext>(
            var_grad.size(),
            static_cast<float>(0),
            mem->mutable_device_data<float>()
            );
        _var_hist[var_grad] = mem;
    }
    math::gradient_descent_momentum<float, CPUContext>(
        var.size(),
        var.mutable_device_data<float>(),
        var_grad.device_data<float>(),
        _var_hist[var_grad]->mutable_device_data<float>(),
        static_cast<float>(_lr),
        static_cast<float>(_mm),
        static_cast<float>(0)
        );
#endif
}

} // end namespace optimizer

namespace functional{

opt::optimizer_ptr create_gradient_descent(double lr, double momentum){
    return std::make_shared<opt::gradient_descent>(lr, momentum);
}

} // end namespace functional
} // end namespace mlfe