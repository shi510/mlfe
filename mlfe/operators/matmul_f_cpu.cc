#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../device_context/cpu_context.h"
#include "../core/device.h"
#include "../utils/assert.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class MatMul : public OpAlgo{
using T = typename Tp::T;
public:
    MatMul(OpAlgoContext *oac) : OpAlgo(oac, "MatMul"){
        y = oac->get_output(0);
        a = y.get_children()[0];
        b = y.get_children()[1];
        trans_a = oac->get_attr<bool>("trans_a");
        trans_b = oac->get_attr<bool>("trans_b");
        if(trans_a && !trans_b){
            m = a.shape()[1];
            n = b.shape()[1];
            k = a.shape()[0];
            runtime_assert(k == b.shape()[0],
                "MatMul Op : Matrix Shape A and B not matches.");
        }
        else if(!trans_a && trans_b){
            m = a.shape()[0];
            n = b.shape()[0];
            k = a.shape()[1];
            runtime_assert(k == b.shape()[1],
                "MatMul Op : Matrix Shape A and B not matches.");
        }
        else if(trans_a && trans_b){
            m = a.shape()[1];
            n = b.shape()[0];
            k = a.shape()[0];
            runtime_assert(k == b.shape()[1],
                "MatMul Op : Matrix Shape A and B not matches.");
        }
        else{
            m = a.shape()[0];
            n = b.shape()[1];
            k = a.shape()[1];
            runtime_assert(k == a.shape()[1],
                "MatMul Op : Matrix Shape A and B not matches.");
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto a_ptr = a.device_data<T>();
        auto b_ptr = b.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();

        math::gemm<T, CPUContext>(trans_a, trans_b,
                                  m, n, k,
                                  T(1), a_ptr, a.shape()[1],
                                  b_ptr, b.shape()[1],
                                  T(0), y_ptr, y.shape()[1], nullptr
                                 );
    }

private:
    Tensor a;
    Tensor b;
    Tensor y;
    bool trans_a, trans_b;
    int m, n, k;
};

REGIST_OP_ALGO(MatMul)
    .Input("A", type::float32::string)
    .Input("B", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MatMul<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
