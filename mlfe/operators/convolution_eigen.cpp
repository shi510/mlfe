#ifndef __CONVOLUTION_EIGEN_OP_HPP__
#define __CONVOLUTION_EIGEN_OP_HPP__
#include <unsupported/Eigen/CXX11/Tensor>
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../math/transform.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "convolution.hpp"

namespace mlfe{

template <class DataType>
class ConvolutionWithEigenOp : public ConvolutionBaseOp<CPUContext>{
public:
    explicit ConvolutionWithEigenOp(
                                    OperatorIO &opio,
                                    ItemHolder *ih
                                    ) : ConvolutionBaseOp<CPUContext>(opio, ih){
        runtime_assert(inputs.size() == 3,
                       "[Convolution With Eigen Op] inputs.size() == 3");
        runtime_assert(outputs.size() == 1,
                       "[Convolution With Eigen Op] outputs.size() == 1");
        const auto x = inputs[InputSchema::x];
        const auto w = inputs[InputSchema::w];
        const auto b = inputs[InputSchema::b];
        auto y = outputs[OutputSchema::y];
        
        if(opio.param.HasParam("Filters") &&
           opio.param.HasParam("Kernel") &&
           w->IsEmpty() &&
           b->IsEmpty() &&
           y->IsEmpty() &&
           !x->IsEmpty() &&
           x->Dims() == 4){
            filters = opio.param.GetParam<int>("Filters");
            kernel_size = opio.param.GetParam<std::vector<int>>("Kernel");
            out_h = OutHeightSize();
            out_w = OutWidthSize();
            w->template Resize<DataType>({filters, x->Dim(1), kernel_size[0], kernel_size[1]});
            b->template Resize<DataType>({filters});
            y->template Resize<DataType>({x->Dim(0), filters, out_h, out_w});
        }
        else{
            runtime_assert(x->Dims() == 4,
                           "[Convolution With Eigen Op] x->Dims() == 4");
            runtime_assert(kernel_size.size() == 2,
                           "[Convolution With Eigen Op] : kernel.size() == 2");
            runtime_assert(w->Dim(0) == b->Size(),
                           "[Convolution With Eigen Op] : filter->Dim(0) == bias->Size()");
        }
    }
    
    void Compute() override{
        const auto x = inputs[InputSchema::x];
        const auto w = inputs[InputSchema::w];
        const auto b = inputs[InputSchema::b];
        auto y = outputs[OutputSchema::y];
        
        Eigen::Tensor<DataType, 4, Eigen::RowMajor> x_t = Eigen::TensorMap<Eigen::Tensor<DataType, 4, Eigen::RowMajor>>(
                                                                                     x->template GetPtrMutable<DataType>(),
                                                                                     x->Dim(0),
                                                                                     x->Dim(1),
                                                                                     x->Dim(2),
                                                                                     x->Dim(3)
                                                                                     ).shuffle(Eigen::array<int, 4>{{0, 2, 3, 1}});
        
        Eigen::Tensor<DataType, 4, Eigen::RowMajor> kernel_t = Eigen::TensorMap<Eigen::Tensor<DataType, 4, Eigen::RowMajor>>(
                                                                                          w->template GetPtrMutable<DataType>(),
                                                                                          w->Dim(0),
                                                                                          w->Dim(1),
                                                                                          w->Dim(2),
                                                                                          w->Dim(3)
                                                                                          ).shuffle(Eigen::array<int, 4>{{2, 3, 1, 0}});
        
        Eigen::Tensor<DataType, 4, Eigen::RowMajor> y_t(y->Dim(0), y->Dim(2), y->Dim(3), y->Dim(1));
        
        y_t = x_t.extract_image_patches(
                                        w->Dim(2),
                                        w->Dim(3),
                                        stride[0], stride[1],
                                        1, 1,
                                        1, 1,
                                        padding, padding,
                                        padding, padding,
                                        0)
        .reshape(Eigen::array<int, 2>{{y->Size() / y->Dim(1), w->Size() / w->Dim(0)}})
        .contract(
                  kernel_t.reshape(Eigen::array<int, 2>{{w->Size() / w->Dim(0), w->Dim(0)}}),
                  Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(1, 0)}}
                  )
        .reshape(y_t.dimensions());
        
        Eigen::Map<Eigen::Array<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> y_arr(y_t.data(), y->Dim(1), y->Size() / y->Dim(1));
        Eigen::Map<Eigen::Array<DataType, Eigen::Dynamic, 1>> b_arr(b->template GetPtrMutable<DataType>(), b->Size(), 1);
        y_arr = y_arr.colwise() + b_arr;
        
        Eigen::TensorMap<Eigen::Tensor<DataType, 4, Eigen::RowMajor>>(
                                                 y->template GetPtrMutable<DataType>(),
                                                 y->Dim(0),
                                                 y->Dim(1),
                                                 y->Dim(2),
                                                 y->Dim(3)
                                                 ) = y_t.shuffle(Eigen::array<int, 4>{{0, 3, 1, 2}});
    }
    
private:
    enum InputSchema{x, w, b};
    enum OutputSchema{y};
    int out_h;
    int out_w;
};

REGIST_OPERATOR_CPU(Conv_float_Eigen, ConvolutionWithEigenOp<float>)
REGIST_OPERATOR_CPU(Conv_double_Eigen, ConvolutionWithEigenOp<double>)

template <class DataType>
class ConvolutionGradientOp : public ConvolutionBaseOp<CPUContext>{
public:
    explicit ConvolutionGradientOp(
                                   OperatorIO &opio,
                                   ItemHolder *ih
                                   ) : ConvolutionBaseOp<CPUContext>(opio, ih){
        runtime_assert(inputs.size() == 3,
                       "[Convolution Gradient Op] inputs.size() == 3");
        runtime_assert(outputs.size() == 3,
                       "[Convolution Gradient Op] outputs.size() == 3");
        const auto x = inputs[InputSchema::x];
        const auto w = inputs[InputSchema::w];
        const auto dy = inputs[InputSchema::dy];
        auto dw = outputs[OutputSchema::dw];
        auto db = outputs[OutputSchema::db];
        auto dx = outputs[OutputSchema::dx];
        
        if(opio.param.HasParam("Filters") &&
           opio.param.HasParam("Kernel") &&
           dw->IsEmpty() &&
           db->IsEmpty() &&
           dx->IsEmpty() &&
           !x->IsEmpty() &&
           !w->IsEmpty() &&
           !dy->IsEmpty()
           ){
            filters = opio.param.GetParam<int>("Filters");
            kernel_size = opio.param.GetParam<std::vector<int>>("Kernel");
            dw->template Resize<DataType>(*w);
            db->template Resize<DataType>({filters});
            dx->template Resize<DataType>(*x);
        }
        else{
            runtime_assert(x->Dims() == 4,
                           "[Convolution Gradient Op] x->Dims() == 4");
            runtime_assert(kernel_size.size() == 2,
                           "[Convolution Gradient Op] Kernel Param Dim must be 2.");
            runtime_assert(w->Size() == dw->Size(),
                           "[Convolution Gradient Op] : w->Size() == dw->Size()");
        }
        
        m = filters;
        n = OutHeightSize() * OutWidthSize();
        k = kernel_size[0] * kernel_size[1] * x->Dim(1);
        
        bias_multiplier.template Resize<DataType, CPUContext>({n});
        math::set<DataType, CPUContext>(
                                        bias_multiplier.Size(),
                                        static_cast<DataType>(1),
                                        bias_multiplier.template GetPtrMutable<DataType>()
                                        );
        
        col_buf.template Resize<DataType, CPUContext>({k, n});
    }
    
    void Compute() override{
        const auto x = inputs[InputSchema::x];
        const auto w = inputs[InputSchema::w];
        const auto dy = inputs[InputSchema::dy];
        auto dw = outputs[OutputSchema::dw];
        auto db = outputs[OutputSchema::db];
        auto dx = outputs[OutputSchema::dx];
        int batch_size = x->Dim(0);
        const DataType *x_ptr = x->template GetPtrConst<DataType>();
        const DataType *dy_ptr = dy->template GetPtrMutable<DataType>();
        DataType *col_ptr = col_buf.template GetPtrMutable<DataType>();
        DataType *dx_ptr = dx->template GetPtrMutable<DataType>();
        
        math::scal<DataType, CPUContext>(
                                         dx->Size(), DataType(0),
                                         dx->template GetPtrConst<DataType>(),
                                         dx->template GetPtrMutable<DataType>()
                                         );
        math::scal<DataType, CPUContext>(
                                         dw->Size(), DataType(0),
                                         dw->template GetPtrConst<DataType>(),
                                         dw->template GetPtrMutable<DataType>()
                                         );
        math::scal<DataType, CPUContext>(
                                         db->Size(), DataType(0),
                                         db->template GetPtrConst<DataType>(),
                                         db->template GetPtrMutable<DataType>()
                                         );
        
        for(int i = 0; i < batch_size; ++i){
            /*
             * gradient w.r.t. bias.
             */
            math::gemv<DataType, CPUContext>(
                                             false, m, n,
                                             DataType(1), dy_ptr, n,
                                             bias_multiplier.template GetPtrConst<DataType>(), DataType(1),
                                             db->template GetPtrMutable<DataType>(), DataType(1), nullptr
                                             );
            
            math::im2col<DataType, CPUContext>(
                                               x->Dim(1), x->Dim(2), x->Dim(3),
                                               kernel_size[0], kernel_size[1],
                                               stride[0], padding,
                                               x_ptr, col_ptr
                                               );
            
            /*
             * Calculate gradients of weights.
             * kernel_size = {kernel_h, kernel_w, channel_of_x} = k
             * filters = {number of feature map channel} = m
             * out_size = {y_h, y_w} = n
             * dy({filters, out_size}) * col({kernel_size, out_size})^T
             *  = dw({filters, kernel_size})
             */
            math::gemm<DataType, CPUContext>(
                                             false, true, m, k, n,
                                             DataType(1), dy_ptr, n,
                                             col_ptr, n,
                                             DataType(1), dw->template GetPtrMutable<DataType>(), k, nullptr
                                             );
            
            /*
             * Calculate loss to propagate through bottom.
             * w({filters, kernel_size})^T * dy({filters, out_size})
             *  = col({kernel_size, out_size})
             */
            math::gemm<DataType, CPUContext>(
                                             true, false, k, n, m,
                                             DataType(1), w->template GetPtrConst<DataType>(), k,
                                             dy_ptr, n,
                                             DataType(0), col_ptr, n, nullptr
                                             );
            
            math::col2im<DataType, CPUContext>(
                                               col_ptr,
                                               x->Dim(1), x->Dim(2), x->Dim(3),
                                               kernel_size[1], stride[0], padding,
                                               dx_ptr
                                               );
            
            /*
             * next batch.
             */
            x_ptr += x->Size() / x->Dim(0);
            dx_ptr += dx->Size() / dx->Dim(0);
            dy_ptr += n * m;
        }
        
        math::scal<DataType, CPUContext>(
                                         db->Size(),
                                         DataType(1) / static_cast<DataType>(batch_size),
                                         db->template GetPtrConst<DataType>(),
                                         db->template GetPtrMutable<DataType>()
                                         );
        
        math::scal<DataType, CPUContext>(
                                         dw->Size(),
                                         DataType(1) / static_cast<DataType>(batch_size),
                                         dw->template GetPtrConst<DataType>(),
                                         dw->template GetPtrMutable<DataType>()
                                         );
    }
private:
    enum InputSchema{x, w, dy};
    enum OutputSchema{dw, db, dx};
    TensorBlob<CPUContext> col_buf;
    TensorBlob<CPUContext> bias_multiplier;
    /*
     * Variables for GEMM.
     */
    int m;
    int n;
    int k;
};

REGIST_OPERATOR_CPU(Conv_float_Gradient, ConvolutionGradientOp<float>)
REGIST_OPERATOR_CPU(Conv_double_Gradient, ConvolutionGradientOp<double>)
    
struct ConvolutionGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type + "_Gradient";
        opio_grad.data_type = opio.data_type;
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.inputs[1]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[1] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[2] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};
    
REGIST_OPERATOR_GRADIENT_IO(Conv, ConvolutionGradientIO)

} /* namespace mlfe */
#endif /* __CONVOLUTION_EIGEN_OP_HPP__ */
