#include <random>
#include <curand.h>
#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions_cuda.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ConstantInitCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override{
        auto val_str = oc->attr->GetParam<std::string>("Value");
        _val = to_value<T>(val_str);
        _x = oc->inputs[0];
    }

    void Run() override{
        math::set<T, D>(
            _x->Size(),
            _val,
            _x->GetPtr<T>()
            );
    }

    Tensor *_x;
    T _val;
};

REGIST_NODE_FUNCTOR(ConstantInit, DataType::F32, Accelerator::CUDA, ConstantInitCudaF<float>)

template <typename T>
struct XavierInitCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto seed_str = oc->attr->GetParam<std::string>("Seed");
        _x = oc->inputs[0];
        T scale_denominator = _x->Dims() > 1 ?
            static_cast<T>(_x->Size()) / _x->Dim(0) : static_cast<T>(_x->Size());
        T scale = std::sqrt(
            static_cast<T>(6) /
            static_cast<T>(scale_denominator)
        );
        _a = -scale;
        _b = scale;
        _seed = to_value<T>(seed_str);
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, _seed);
    }

    void Run() override {
        T *x_ptr = _x->GetPtr<T>();
        math::UniformCurand<T>(&gen, _x->Size(), x_ptr, _a, _b);
    }

    ~XavierInitCudaF() {
        cudaFree(_states);
    }

    Tensor *_x;
    T _a, _b;
    int _seed;
    curandGenerator_t gen;
};

REGIST_NODE_FUNCTOR(XavierInit, DataType::F32, Accelerator::CUDA, XavierInitCudaF<float>)

} // end namespace node
} // end namespace mlfe
