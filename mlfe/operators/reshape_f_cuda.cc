#include "../core/op_algo.h"
#include "../core/device.h"
#include <stdexcept>
#include <numeric>

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class Reshape : public OpAlgo{
using T = typename Tp::T;
public:
    Reshape(OpAlgoContext *oac) : OpAlgo(oac, "Reshape"){
        x = oac->get_input(0);
        shape = oac->get_input(1);
        y = oac->get_output(0);
        resize();
    }

    void resize() override {
        auto x_shape = x.shape();
        auto x_size = x.size();
        int sum = 1;
        std::vector<int> new_shape;
        int neg_axis = -1;
        for (int n = 0; n < shape.size(); ++n) {
            auto d = shape.data<int64_t>()[n];
            if (d == -1 && neg_axis > -1) {
                throw std::runtime_error("Reshape operator: wrong shape");
            }
            else if (d == -1) {
                neg_axis = n;
            }
            new_shape.push_back(d);
        }
        if (neg_axis != -1) {
            sum = std::accumulate(new_shape.begin(), new_shape.begin() + neg_axis,
                sum, std::multiplies<int>());
            sum = std::accumulate(new_shape.begin() + neg_axis + 1, new_shape.end(), sum,
                std::multiplies<int>());
            new_shape[neg_axis] = x_size / sum;
        }
        y.resize(new_shape);
    }

    void Compute(op_algo_runtime_context& rc) override{}
private:
    Tensor x;
    Tensor shape;
    Tensor y;
};

REGIST_OP_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Reshape<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReshapeGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReshapeGrad(OpAlgoContext *oac) 
        : OpAlgo(oac, "ReshapeGradient"){}

    void Compute(op_algo_runtime_context& rc) override{}

private:
    Tensor dy;
    Tensor dx;
};

REGIST_OP_GRAD_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReshapeGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
