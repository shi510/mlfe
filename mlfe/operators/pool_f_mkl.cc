#include "../core/op_algo.h"
#include "../core/device.h"
#include "third_party/mkldnn/include/mkldnn.hpp"
#include "third_party/mkldnn/src/common/stream.hpp"
#include "third_party/mkldnn/src/common/event.hpp"

namespace mlfe{
namespace algorithm_mkl{
using namespace mkldnn;
    
// TODO: Do not use mkldnn api, but internal algorithm.
template <class Tp>
class MaxPool : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPool(OpAlgoContext *oac) : OpAlgo(oac, "MaxPool"){
        using data_type = mkldnn::memory::data_type;
        using data_order = mkldnn::memory::format;
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_algo = pooling_forward;
        using fwd_desc = pooling_forward::desc;
        using fwd_prim_desc = pooling_forward::primitive_desc;
        using IntVec = std::vector<type::int32::T>;
        auto make_smem = [](const mem_prim_desc arg, void *ptr = nullptr){
            if(ptr == nullptr){
                return std::make_shared<mkldnn::memory>(arg);
            }
            else{
                return std::make_shared<mkldnn::memory>(arg, ptr);
            }
        };
        auto strides = oac->get_attr<IntVec>("stride");
        auto padding = oac->get_attr<IntVec>("padding");
        auto filters_hw = oac->get_attr<IntVec>("kernel");
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
        x = y.get_children()[0];
        idx = oac->get_attr<Tensor>("idx");
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);
        
        x_memory = make_smem(make_mem_prim_desc(x.shape(), data_order::nchw),
                             const_cast<T *>(x.data<T>())
                             );
        
        y_memory = make_smem(make_mem_prim_desc(y.shape(), data_order::nchw),
                             y.mutable_data<T>());
        
        auto x_md = mem_desc(x.shape(), data_type::f32, data_order::nchw);
        auto y_md = mem_desc(y.shape(), data_type::f32, data_order::nchw);
        
        auto desc = fwd_desc(prop_kind::forward,
                             algorithm::pooling_max, x_md,
                             y_md, strides, filters_hw,
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
        
        y_memory_reorder = make_smem(prim_desc.dst_primitive_desc());
        ws_mem = make_smem(prim_desc.workspace_primitive_desc());
        
        net.push_back(fwd_algo(prim_desc,
                               *x_memory_reorder,
                               *y_memory_reorder,
                               *ws_mem
                               )
                      );
        
        if (y_memory_reorder != y_memory){
            net.push_back(reorder(*y_memory_reorder, *y_memory));
        }
        y.get_context().add_attr({"mkldnn_ws", ws_mem});
    }

    void Compute(op_algo_runtime_context& rc) override{
        for(auto &op : net){
            op.get()->execute(&e);
        }
    }
private:
    Tensor x;
    Tensor idx;
    Tensor y;
    impl::event_t e;
    std::vector<primitive> net;
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<mkldnn::memory> x_memory;
    std::shared_ptr<mkldnn::memory> x_memory_reorder;
    std::shared_ptr<mkldnn::memory> y_memory;
    std::shared_ptr<mkldnn::memory> y_memory_reorder;
    std::shared_ptr<mkldnn::memory> ws_mem;
};

REGIST_OP_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Output("IDX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class MaxPoolGrad : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPoolGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using data_type = mkldnn::memory::data_type;
        using data_order = mkldnn::memory::format;
        using mem_desc = mkldnn::memory::desc;
        using mem_prim_desc = mkldnn::memory::primitive_desc;
        using fwd_desc = pooling_forward::desc;
        using fwd_prim_desc = pooling_forward::primitive_desc;
        using bwd_algo = pooling_backward;
        using bwd_desc = pooling_backward::desc;
        using bwd_prim_desc = pooling_backward::primitive_desc;
        using IntVec = std::vector<type::int32::T>;
        auto strides = oac->get_attr<IntVec>("stride");
        auto padding = oac->get_attr<IntVec>("padding");
        auto filters_hw = oac->get_attr<IntVec>("kernel");
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
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        y = dx.get_children()[1];
        dy = dx.get_children()[2];
        ws_mem = y.get_context().template get_attr
        <std::shared_ptr<mkldnn::memory>>("mkldnn_ws");
        cpu_engine = std::make_shared<engine>(engine::cpu, 0);
        
        dx_memory = make_smem(make_mem_prim_desc(dx.shape(), data_order::nchw),
                              dx.mutable_device_data<T>()
                              );
        
        dy_memory = make_smem(make_mem_prim_desc(dy.shape(), data_order::nchw),
                              dy.mutable_device_data<T>());
        
        auto dx_md = mem_desc(dx.shape(), data_type::f32, data_order::nchw);
        auto dy_md = mem_desc(dy.shape(), data_type::f32, data_order::nchw);
        
        auto desc = bwd_desc(algorithm::pooling_max, dx_md,
                             dy_md, strides, filters_hw,
                             padding, padding,
                             padding_kind::zero);
        
        auto hint_desc = fwd_desc(prop_kind::forward,
                                  algorithm::pooling_max, dx_md,
                                  dy_md, strides, filters_hw,
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
        
        dx_memory_reorder = make_smem(prim_desc.diff_src_primitive_desc());
        
        net.push_back(bwd_algo(prim_desc,
                               *dy_memory_reorder,
                               *ws_mem,
                               *dx_memory_reorder
                               )
                      );
        
        if (dx_memory_reorder != dx_memory){
            net.push_back(reorder(*dx_memory_reorder, *dx_memory));
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        for(auto &op : net){
            op.get()->execute(&e);
        }
    }

private:
    Tensor x;
    Tensor y;
    Tensor dy;
    Tensor dx;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    impl::event_t e;
    std::vector<primitive> net;
    std::shared_ptr<engine> cpu_engine;
    std::shared_ptr<mkldnn::memory> dx_memory;
    std::shared_ptr<mkldnn::memory> dx_memory_reorder;
    std::shared_ptr<mkldnn::memory> dy_memory;
    std::shared_ptr<mkldnn::memory> dy_memory_reorder;
    std::shared_ptr<mkldnn::memory> ws_mem;
};

REGIST_OP_GRAD_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Input("IDX", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU(MKLDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_mkl
} // end namespace mlfe
