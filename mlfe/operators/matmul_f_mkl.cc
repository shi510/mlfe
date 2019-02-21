#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/utils/assert.h"
#include "third_party/mkldnn/external/mklml_mac_2019.0.1.20180928/include/mkl_cblas.h"

namespace mlfe{
namespace algorithm_mkl{

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
            k = b.shape()[1];
            runtime_assert(k == a.shape()[1],
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

    void Compute() override{
        auto a_ptr = a.device_data<T>();
        auto b_ptr = b.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        const float alpha = T(1);
        const float beta = T(0);
        const int lda = trans_a ? m : k;
        const int ldb = trans_b ? k : n;
        const int ldc = n;

        cblas_sgemm(CblasRowMajor,
                    trans_a ? CblasTrans : CblasNoTrans,
                    trans_b ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    alpha, a_ptr, lda,
                    b_ptr, ldb, beta,
                    y_ptr, ldc);
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
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MatMul<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_mkl
} // end namespace mlfe
