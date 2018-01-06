#ifndef __CONVOLUTION_EIGEN_OP_HPP__
#define __CONVOLUTION_EIGEN_OP_HPP__
#include <unsupported/Eigen/CXX11/Tensor>
#include "../math/blas.hpp"
#include "../math/transform.hpp"
#include "../utils/assert.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "convolution_op.hpp"

namespace mlfe{

template <class DataType>
class ConvolutionWithEigenOp : public ConvolutionBaseOp<CPUContext>{
public:
    explicit ConvolutionWithEigenOp(
                                    std::vector<std::shared_ptr<TensorBlob<CPUContext>>> inputs,
                                    std::vector<std::shared_ptr<TensorBlob<CPUContext>>> outputs,
                                    ParamDef param
                                    ) : ConvolutionBaseOp<CPUContext>(inputs, outputs, param){
        const auto x = this->Input(InputSchema::x);
        const auto w = this->Input(InputSchema::w);
        const auto b = this->Input(InputSchema::b);
        auto y = this->Output(OutputSchema::y);
        
        if(!GetParam().GetParamByName("Stride", stride)){
            stride = {1, 1};
        }
        if(!GetParam().GetParamByName("Padding", padding)){
            padding = 0;
        }
        
        if(GetParam().GetParamByName("Filters", filters) &&
           GetParam().GetParamByName("Kernel", kernel_size) &&
           w->IsEmpty() &&
           b->IsEmpty() &&
           y->IsEmpty() &&
           !x->IsEmpty() &&
           x->Dims() == 4){
            out_h = OutHeightSize();
            out_w = OutWidthSize();
            w->template Resize<DataType>({filters, x->Dim(1), kernel_size[0], kernel_size[1]});
            b->template Resize<DataType>({filters});
            y->template Resize<DataType>({x->Dim(0), filters, out_h, out_w});
        }
        else{
            runtime_assert(x->Dims() == 4, "ConvolutionOp::Setup() : Input x's demension size must be 4.");
            runtime_assert(GetParam().GetParamByName("Filters", filters), "ConvolutionOp::Setup() : Filter size can not find.");
            runtime_assert(GetParam().GetParamByName("Kernel", kernel_size), "ConvolutionOp::Setup() : Kernel can not find.");
            runtime_assert(kernel_size.size() == 2, "ConvolutionOp::Setup() : Kernel Param Dim must be 2.");
            runtime_assert(w->Dim(0) == b->Size(), "ConvolutionOp::Setup() : filter's 0 dim must be same bias's size.");
        }
    }
    
    void Compute() override{
        using namespace Eigen;
        const auto x = Input(InputSchema::x);
        const auto w = Input(InputSchema::w);
        const auto b = Input(InputSchema::b);
        auto y = Output(OutputSchema::y);
        Tensor<DataType, 4, RowMajor> x_t = TensorMap<Tensor<DataType, 4, RowMajor>>(
                                                                                     x->template GetPtrMutable<DataType>(),
                                                                                     x->Dim(0),
                                                                                     x->Dim(1),
                                                                                     x->Dim(2),
                                                                                     x->Dim(3)
                                                                                     ).shuffle(Eigen::array<int, 4>{{0, 2, 3, 1}});
        
        Tensor<DataType, 4, RowMajor> kernel_t = TensorMap<Tensor<DataType, 4, RowMajor>>(
                                                                                          w->template GetPtrMutable<DataType>(),
                                                                                          w->Dim(0),
                                                                                          w->Dim(1),
                                                                                          w->Dim(2),
                                                                                          w->Dim(3)
                                                                                          ).shuffle(Eigen::array<int, 4>{{2, 3, 1, 0}});
        
        Tensor<DataType, 4, RowMajor> y_t(y->Dim(0), y->Dim(2), y->Dim(3), y->Dim(1));
        
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
                  Eigen::array<IndexPair<int>, 1>{{IndexPair<int>(1, 0)}}
                  )
        .reshape(y_t.dimensions());
        
        Map<Array<DataType, Dynamic, Dynamic, RowMajor>> y_arr(y_t.data(), y->Dim(1), y->Size() / y->Dim(1));
        Map<Array<DataType, Dynamic, 1>> b_arr(b->template GetPtrMutable<DataType>(), b->Size(), 1);
        y_arr = y_arr.colwise() + b_arr;
        
        TensorMap<Tensor<DataType, 4, RowMajor>>(
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

template <class DataType>
class ConvolutionGradientWithEigenOp : public ConvolutionBaseOp<CPUContext>{
public:
    explicit ConvolutionGradientWithEigenOp(
                                            std::vector<std::shared_ptr<TensorBlob<CPUContext>>> inputs,
                                            std::vector<std::shared_ptr<TensorBlob<CPUContext>>> outputs,
                                            ParamDef param
                                            ) : ConvolutionBaseOp<CPUContext>(inputs, outputs, param){
        const auto x = this->Input(InputSchema::x);
        const auto w = this->Input(InputSchema::w);
        const auto dy = this->Input(InputSchema::dy);
        auto dw = this->Output(OutputSchema::dw);
        auto db = this->Output(OutputSchema::db);
        auto dx = this->Output(OutputSchema::dx);
        int out_h, out_w;
        
        runtime_assert(x->Dims() == 4, "ConvolutionOp::Setup() : Input x's demension size must be 4.");
        runtime_assert(GetParam().GetParamByName("Filters", filters), "ConvolutionOp::Setup() : Filter size can not find.");
        runtime_assert(GetParam().GetParamByName("Kernel", kernel_size), "ConvolutionOp::Setup() : Kernel can not find.");
        runtime_assert(kernel_size.size() == 2, "ConvolutionOp::Setup() : Kernel Param Dim must be 2.");
        
        if(!GetParam().GetParamByName("Stride", stride)){
            stride = {1, 1};
        }
        if(!GetParam().GetParamByName("Padding", padding)){
            padding = 0;
        }
        
        if(dw->IsEmpty() &&
           db->IsEmpty() &&
           dx->IsEmpty() &&
           !x->IsEmpty() &&
           !w->IsEmpty() &&
           !dy->IsEmpty()
           ){
            dw->template Resize<DataType>(w);
            db->template Resize<DataType>({filters});
            dx->template Resize<DataType>(x);
        }
        
        out_h = OutHeightSize();
        out_w = OutWidthSize();
        
        m = filters;
        n = out_h * out_w;
        k = kernel_size[0] * kernel_size[1] * x->Dim(1);
        
        bias_multiplier.Resize<DataType, CPUContext>({n});
        bias_multiplier.SetByConst<DataType>(DataType(1));
        
        col_buf.Resize<DataType, CPUContext>({k, n});
    }
    
    void Compute() override{
        const auto x = Input(InputSchema::x);
        const auto w = Input(InputSchema::w);
        const auto dy = Input(InputSchema::dy);
        auto dw = Output(OutputSchema::dw);
        auto db = Output(OutputSchema::db);
        auto dx = Output(OutputSchema::dx);
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

} /* namespace mlfe */
#endif /* __CONVOLUTION_EIGEN_OP_HPP__ */
