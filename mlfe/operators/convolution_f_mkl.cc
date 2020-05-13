#include "../core/op_algo.h"
#include "../core/device.h"
#include "third_party/mkldnn/include/mkldnn.hpp"
#include "third_party/mkldnn/src/common/stream.hpp"
#include "third_party/mkldnn/src/common/event.hpp"
#include <memory>

namespace mlfe{
namespace algorithm_mkl{
using namespace mkldnn;

// TODO: Do not use mkldnn api, but internal algorithm.
template <class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
using IntVec = std::vector<type::int32::T>;
public:
    Convolution(OpAlgoContext *oac) : OpAlgo(oac, "Convolution"){
        using data_type = mkldnn::memory::data_type;
        using data_order = mkldnn::memory::format;
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_algo = convolution_forward;
        using fwd_desc = convolution_forward::desc;
        using fwd_prim_desc = convolution_forward::primitive_desc;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        auto strides = oac->get_attr<IntVec>("strides");
        auto padding = oac->get_attr<IntVec>("pads");
        auto make_mem_prim_desc = [this](IntVec shape,
                                         data_order order
                                         )
        {
            return mem_prim_desc{
                {
                    shape,
                    data_type::f32,
                    order
                },
                *cpu_engine
            };
        };
        y = oac->get_output(0);
        x = oac->get_input(0);
        w = oac->get_input(1);
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);

        x_memory = make_smem(make_mem_prim_desc(x.shape(), data_order::nchw),
                             x.mutable_data<T>());

        w_memory = make_smem(make_mem_prim_desc(w.shape(),data_order::oihw),
                             w.mutable_data<T>());

        y_memory = make_smem(make_mem_prim_desc(y.shape(), data_order::nchw),
                             y.mutable_data<T>());

        auto x_md = mem_desc(x.shape(), data_type::f32, data_order::any);
        auto w_md = mem_desc(w.shape(), data_type::f32, data_order::any);
        auto y_md = mem_desc(y.shape(), data_type::f32, data_order::any);

        auto desc = fwd_desc(prop_kind::forward_inference,
                             convolution_direct, x_md,
                             w_md, y_md, strides,
                             padding, padding,
                             padding_kind::zero);

        auto prim_desc = fwd_prim_desc(desc, *cpu_engine);

        x_memory_reorder = x_memory;
        if (mem_prim_desc(prim_desc.src_primitive_desc()) !=
            x_memory->get_primitive_desc())
        {
            x_memory_reorder = make_smem(prim_desc.src_primitive_desc());
            net.push_back(reorder(*x_memory, *x_memory_reorder));
        }

        w_memory_reorder = w_memory;
        if (mem_prim_desc(prim_desc.weights_primitive_desc()) !=
            w_memory->get_primitive_desc())
        {
            w_memory_reorder = make_smem(prim_desc.weights_primitive_desc());
            net.push_back(reorder(*w_memory, *w_memory_reorder));
        }
        y_memory_reorder = make_smem(prim_desc.dst_primitive_desc());

        net.push_back(fwd_algo(prim_desc,
                               *x_memory_reorder,
                               *w_memory_reorder,
                               *y_memory_reorder
                               )
                      );

        if (y_memory_reorder != y_memory){
            net.push_back(reorder(*y_memory_reorder, *y_memory));
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        for(auto &op : net){
            op.get()->execute(&e);
        }
    }

private:
    Tensor x;
    Tensor w;
    Tensor y;
    impl::event_t e;
    std::vector<primitive> net;
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<mkldnn::memory> x_memory;
    std::shared_ptr<mkldnn::memory> x_memory_reorder;
    std::shared_ptr<mkldnn::memory> w_memory;
    std::shared_ptr<mkldnn::memory> w_memory_reorder;
    std::shared_ptr<mkldnn::memory> y_memory;
    std::shared_ptr<mkldnn::memory> y_memory_reorder;
};

REGIST_OP_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

// TODO: Do not use mkldnn api, but internal algorithm.
template <class Tp>
class Conv2DGradientInput : public OpAlgo{
using T = typename Tp::T;
public:
    Conv2DGradientInput(OpAlgoContext *oac) : OpAlgo(oac){
        using data_type = mkldnn::memory::data_type;
        using data_order = mkldnn::memory::format;
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_desc = convolution_forward::desc;
        using fwd_prim_desc = convolution_forward::primitive_desc;
        using bwd_algo = convolution_backward_data;
        using bwd_desc = convolution_backward_data::desc;
        using bwd_prim_desc = convolution_backward_data::primitive_desc;
        using IntVec = std::vector<type::int32::T>;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        auto make_mem_prim_desc = [this](IntVec shape,
                                         data_order order
                                         )
        {
            return mem_prim_desc{
                {
                    shape,
                    data_type::f32,
                    order
                },
                *cpu_engine
            };
        };
        auto strides = oac->get_attr<IntVec>("strides");
        auto padding = oac->get_attr<IntVec>("pads");
        dx = oac->get_output(0);
        w = oac->get_input(0);
        dy = oac->get_input(1);
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);

        dx_memory = make_smem(make_mem_prim_desc(dx.shape(), data_order::nchw),
                             dx.mutable_data<T>());

        w_memory = make_smem(make_mem_prim_desc(w.shape(), data_order::oihw),
                            const_cast<T *>(w.data<T>()));

        dy_memory = make_smem(make_mem_prim_desc(dy.shape(), data_order::nchw),
                             const_cast<T *>(dy.data<T>()));

        auto dx_md = mem_desc(dx.shape(), data_type::f32, data_order::any);
        auto w_md = mem_desc(w.shape(), data_type::f32, data_order::any);
        auto dy_md = mem_desc(dy.shape(), data_type::f32, data_order::any);
        auto desc = bwd_desc(convolution_direct, dx_md,
                             w_md, dy_md, strides,
                             padding, padding,
                             padding_kind::zero);

        auto hint_desc = fwd_desc(prop_kind::forward_inference,
                                  convolution_direct, dx_md,
                                  w_md, dy_md, strides,
                                  padding, padding,
                                  padding_kind::zero);

        auto hint_prim_desc = fwd_prim_desc(hint_desc, *cpu_engine);

        auto prim_desc = bwd_prim_desc(desc, *cpu_engine, hint_prim_desc);

        dy_memory_reorder = dy_memory;
        if (mem_prim_desc(prim_desc.diff_dst_primitive_desc()) !=
            dy_memory->get_primitive_desc())
        {
            dy_memory_reorder = make_smem(prim_desc.diff_dst_primitive_desc());
            net.push_back(reorder(*dy_memory, *dy_memory_reorder));
        }

        w_memory_reorder = w_memory;
        if (mem_prim_desc(prim_desc.weights_primitive_desc()) !=
            w_memory->get_primitive_desc())
        {
            w_memory_reorder = make_smem(prim_desc.weights_primitive_desc());
            net.push_back(reorder(*w_memory, *w_memory_reorder));
        }
        dx_memory_reorder = make_smem(prim_desc.diff_src_primitive_desc());

        net.push_back(bwd_algo(prim_desc,
                               *dy_memory_reorder,
                               *w_memory_reorder,
                               *dx_memory_reorder
                               )
                      );

        if (dx_memory_reorder != dx_memory) {
            net.push_back(reorder(*dx_memory_reorder, *dx_memory));
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        for(auto &op : net){
            op.get()->execute(&e);
        }
    }

private:
    Tensor w;
    Tensor dx;
    Tensor dy;
    impl::event_t e;
    std::vector<primitive> net;
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<mkldnn::memory> dx_memory;
    std::shared_ptr<mkldnn::memory> dx_memory_reorder;
    std::shared_ptr<mkldnn::memory> w_memory;
    std::shared_ptr<mkldnn::memory> w_memory_reorder;
    std::shared_ptr<mkldnn::memory> dy_memory;
    std::shared_ptr<mkldnn::memory> dy_memory_reorder;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientInput)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientInput<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

// TODO: Do not use mkldnn api, but internal algorithm.
template <class Tp>
class Conv2DGradientFilter : public OpAlgo{
using T = typename Tp::T;
public:
    Conv2DGradientFilter(OpAlgoContext *oac) : OpAlgo(oac){
        using data_type = mkldnn::memory::data_type;
        using data_order = mkldnn::memory::format;
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_desc = convolution_forward::desc;
        using fwd_prim_desc = convolution_forward::primitive_desc;
        using bwd_algo = convolution_backward_weights;
        using bwd_desc = convolution_backward_weights::desc;
        using bwd_prim_desc = convolution_backward_weights::primitive_desc;
        using IntVec = std::vector<type::int32::T>;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        auto make_mem_prim_desc = [this](IntVec shape,
                                         data_order order
                                         )
        {
            return mem_prim_desc{
                {
                    shape,
                    data_type::f32,
                    order
                },
                *cpu_engine
            };
        };
        auto strides = oac->get_attr<IntVec>("strides");
        auto padding = oac->get_attr<IntVec>("pads");
        dw = oac->get_output(0);
        x = oac->get_input(0);
        dy = oac->get_input(1);
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);

        x_memory = make_smem(make_mem_prim_desc(x.shape(), data_order::nchw),
                             const_cast<T *>(x.mutable_data<T>()));

        dw_memory = make_smem(make_mem_prim_desc(dw.shape(), data_order::oihw),
                              dw.mutable_data<T>());

        dy_memory = make_smem(make_mem_prim_desc(dy.shape(), data_order::nchw),
                              const_cast<T *>(dy.data<T>()));

        auto x_md = mem_desc(x.shape(), data_type::f32, data_order::any);
        auto dw_md = mem_desc(dw.shape(), data_type::f32, data_order::any);
        auto dy_md = mem_desc(dy.shape(), data_type::f32, data_order::any);

        auto desc = bwd_desc(convolution_direct, x_md,
                             dw_md, dy_md, strides,
                             padding, padding,
                             padding_kind::zero);

        auto hint_desc = fwd_desc(prop_kind::forward_inference,
                                  convolution_direct, x_md,
                                  dw_md, dy_md, strides,
                                  padding, padding,
                                  padding_kind::zero);

        auto hint_prim_desc = fwd_prim_desc(hint_desc, *cpu_engine);

        auto prim_desc = bwd_prim_desc(desc, *cpu_engine, hint_prim_desc);

        dy_memory_reorder = dy_memory;
        if (mem_prim_desc(prim_desc.diff_dst_primitive_desc()) !=
            dy_memory->get_primitive_desc())
        {
            dy_memory_reorder = make_smem(prim_desc.diff_dst_primitive_desc());
            net.push_back(reorder(*dy_memory, *dy_memory_reorder));
        }

        x_memory_reorder = x_memory;
        if (mem_prim_desc(prim_desc.src_primitive_desc()) !=
            x_memory->get_primitive_desc())
        {
            x_memory_reorder = make_smem(prim_desc.src_primitive_desc());
            net.push_back(reorder(*x_memory, *x_memory_reorder));
        }
        dw_memory_reorder = make_smem(prim_desc.diff_weights_primitive_desc());

        net.push_back(bwd_algo(prim_desc,
                               *x_memory_reorder,
                               *dy_memory_reorder,
                               *dw_memory_reorder
                               )
                      );

        if (dw_memory_reorder != dw_memory) {
            net.push_back(reorder(*dw_memory_reorder, *dw_memory));
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        for(auto &op : net){
            op.get()->execute(&e);
        }
    }

private:
    Tensor x;
    Tensor dy;
    Tensor dw;
    impl::event_t e;
    std::vector<primitive> net;
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<mkldnn::memory> x_memory;
    std::shared_ptr<mkldnn::memory> x_memory_reorder;
    std::shared_ptr<mkldnn::memory> dw_memory;
    std::shared_ptr<mkldnn::memory> dw_memory_reorder;
    std::shared_ptr<mkldnn::memory> dy_memory;
    std::shared_ptr<mkldnn::memory> dy_memory_reorder;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientFilter)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientFilter<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_mkl
} // end namespace mlfe
