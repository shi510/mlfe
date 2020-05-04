#include "mlfe/core/op_algo.h"
#include "third_party/mkldnn/include/mkldnn.hpp"
#include "third_party/mkldnn/src/common/stream.hpp"
#include "third_party/mkldnn/src/common/event.hpp"

namespace mlfe{
namespace algorithm_mkl{
using namespace mkldnn;

// TODO: Do not use mkldnn api, but internal algorithm.
template <class Tp>
class ReLU : public OpAlgo{
using T = typename Tp::T;
public:
    ReLU(OpAlgoContext *oac) : OpAlgo(oac, "ReLU"){
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_algo = eltwise_forward;
        using fwd_desc = eltwise_forward::desc;
        using fwd_prim_desc = eltwise_forward::primitive_desc;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        y = oac->get_output(0);
        x = y.get_children()[0];
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);

        auto x_mem_prim_desc = mem_prim_desc({
            {
                { x.size() },
                mkldnn::memory::data_type::f32,
                mkldnn::memory::format::x
            },
            *cpu_engine
        });

        x_mem = make_smem(x_mem_prim_desc,
                          const_cast<T *>(x.device_data<T>()));

        y_mem = make_smem(x_mem_prim_desc,
                          y.mutable_device_data<T>());

        auto relu_desc = fwd_desc(prop_kind::forward_inference,
                                  algorithm::eltwise_relu,
                                  y_mem->get_primitive_desc().desc(),
                                  negative_slope);

        auto relu_prim_desc = fwd_prim_desc(relu_desc, *cpu_engine);

        algo = std::make_shared<fwd_algo>(relu_prim_desc, *x_mem, *y_mem);
    }

    void Compute(op_algo_runtime_context& rc) override{
        algo->get()->execute(&e);
    }

private:
    Tensor x;
    Tensor y;
    impl::event_t e;
    const float negative_slope = T(0);
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<eltwise_forward> algo;
    std::shared_ptr<mkldnn::memory> x_mem;
    std::shared_ptr<mkldnn::memory> y_mem;
};

REGIST_OP_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLU<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReLUGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReLUGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        dy = dx.get_children()[2];
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_desc = eltwise_forward::desc;
        using fwd_prim_desc = eltwise_forward::primitive_desc;
        using bwd_algo = eltwise_backward;
        using bwd_desc = eltwise_backward::desc;
        using bwd_prim_desc = eltwise_backward::primitive_desc;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);
        
        auto x_mem_prim_desc = mem_prim_desc({
            {
                { x.size() },
                mkldnn::memory::data_type::f32,
                mkldnn::memory::format::x
            },
            *cpu_engine
        });
        
        x_mem = make_smem(x_mem_prim_desc,
                          const_cast<T *>(x.device_data<T>()));
        
        dx_mem = make_smem(x_mem_prim_desc,
                           dx.mutable_device_data<T>());
        
        dy_mem = make_smem(x_mem_prim_desc,
                           dy.mutable_device_data<T>());
        
        auto relu_desc = bwd_desc(algorithm::eltwise_relu,
                                  dy_mem->get_primitive_desc().desc(),
                                  x_mem->get_primitive_desc().desc(),
                                  negative_slope);
        
        auto hint_desc = fwd_desc(prop_kind::forward_inference,
                                  algorithm::eltwise_relu,
                                  x_mem->get_primitive_desc().desc(),
                                  negative_slope);
        
        auto hint_prim_desc = fwd_prim_desc(hint_desc, *cpu_engine);
        
        auto relu_prim_desc = bwd_prim_desc(relu_desc,
                                            *cpu_engine,
                                            hint_prim_desc
                                            );
        
        algo = std::make_shared<bwd_algo>(relu_prim_desc,
                                          *x_mem,
                                          *dy_mem,
                                          *dx_mem
                                          );
    }

    void Compute(op_algo_runtime_context& rc) override{
        algo->get()->execute(&e);
    }

private:
    Tensor x;
    Tensor dy;
    Tensor dx;
    impl::event_t e;
    const float negative_slope = T(0);
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<eltwise_backward> algo;
    std::shared_ptr<mkldnn::memory> x_mem;
    std::shared_ptr<mkldnn::memory> dx_mem;
    std::shared_ptr<mkldnn::memory> dy_mem;
};

REGIST_OP_GRAD_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU(MKL)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLUGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();


template <class Tp>
class Sigmoid : public OpAlgo{
using T = typename Tp::T;
public:
    Sigmoid(OpAlgoContext *oac) : OpAlgo(oac, "Sigmoid"){
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_algo = eltwise_forward;
        using fwd_desc = eltwise_forward::desc;
        using fwd_prim_desc = eltwise_forward::primitive_desc;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        y = oac->get_output(0);
        x = y.get_children()[0];
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);
        
        auto x_mem_prim_desc = mem_prim_desc({
            {
                { x.size() },
                mkldnn::memory::data_type::f32,
                mkldnn::memory::format::x
            },
            *cpu_engine
        });
        
        x_mem = make_smem(x_mem_prim_desc,
                          const_cast<T *>(x.device_data<T>()));
        
        y_mem = make_smem(x_mem_prim_desc,
                          y.mutable_device_data<T>());
        
        auto sigmoid_desc = fwd_desc(prop_kind::forward_inference,
                                     algorithm::eltwise_logistic,
                                     y_mem->get_primitive_desc().desc(),
                                     bias);
        
        auto sigmoid_prim_desc = fwd_prim_desc(sigmoid_desc, *cpu_engine);
        
        algo = std::make_shared<fwd_algo>(sigmoid_prim_desc, *x_mem, *y_mem);
    }
    
    void Compute(op_algo_runtime_context& rc) override{
        algo->get()->execute(&e);
    }
    
private:
    Tensor x;
    Tensor y;
    impl::event_t e;
    const float bias = T(0);
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<eltwise_forward> algo;
    std::shared_ptr<mkldnn::memory> x_mem;
    std::shared_ptr<mkldnn::memory> y_mem;
};

REGIST_OP_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU(MKL)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Sigmoid<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SigmoidGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        dy = dx.get_children()[2];
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_desc = eltwise_forward::desc;
        using fwd_prim_desc = eltwise_forward::primitive_desc;
        using bwd_algo = eltwise_backward;
        using bwd_desc = eltwise_backward::desc;
        using bwd_prim_desc = eltwise_backward::primitive_desc;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);
        auto x_mem_prim_desc = mem_prim_desc({
            {
                { x.size() },
                mkldnn::memory::data_type::f32,
                mkldnn::memory::format::x
            },
            *cpu_engine
        });
        
        x_mem = make_smem(x_mem_prim_desc,
                          const_cast<T *>(x.device_data<T>()));
        
        dx_mem = make_smem(x_mem_prim_desc,
                           dx.mutable_device_data<T>());
        
        dy_mem = make_smem(x_mem_prim_desc,
                           dy.mutable_device_data<T>());
        
        auto sigmoid_desc = bwd_desc(algorithm::eltwise_logistic,
                                     dy_mem->get_primitive_desc().desc(),
                                     x_mem->get_primitive_desc().desc(),
                                     bias);
        
        auto hint_desc = fwd_desc(prop_kind::forward_inference,
                                  algorithm::eltwise_logistic,
                                  x_mem->get_primitive_desc().desc(),
                                  bias);
        
        auto hint_prim_desc = fwd_prim_desc(hint_desc, *cpu_engine);
        
        auto sigmoid_prim_desc = bwd_prim_desc(sigmoid_desc,
                                               *cpu_engine,
                                               hint_prim_desc
                                               );
        
        algo = std::make_shared<bwd_algo>(sigmoid_prim_desc,
                                          *x_mem,
                                          *dy_mem,
                                          *dx_mem
                                          );
    }
    
    void Compute(op_algo_runtime_context& rc) override{
        algo->get()->execute(&e);
    }

private:
    Tensor x;
    Tensor y;
    Tensor dy;
    Tensor dx;
    impl::event_t e;
    const float bias = T(0);
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<eltwise_backward> algo;
    std::shared_ptr<mkldnn::memory> x_mem;
    std::shared_ptr<mkldnn::memory> dx_mem;
    std::shared_ptr<mkldnn::memory> dy_mem;
};

REGIST_OP_GRAD_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU(MKL)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mkl
} // end namespace mlfe
