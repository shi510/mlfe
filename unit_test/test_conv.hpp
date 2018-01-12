#include <iostream>
#include <random>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/core/param_def.hpp>
#include <gtest/gtest.h>
#include <mlfe/utils/gradient_checker.hpp>
#include <mlfe/operators/cast.hpp>

using namespace std;
using namespace mlfe;

TEST(ConvolutionOperatorTest, VerifyCPUResults) {
    ItemHolder ih;
    OperatorIO opio, opio_grad;
    std::shared_ptr<OperatorBase> conv, conv_grad;
    TensorBlob<CPUContext> *x, *w, *b, *y, *dy, *dw, *db, *dx;
    const std::vector<int> kernel_shape = {3, 3};
    const std::vector<int> stride_shape = {1, 1};
    const int batch = 2;
    const int out_filters = 2;
    const int padding = 0;
    const double acceptable_gradient_check_val = 1e-7;
    auto set = [](double *ptr, int from, int to, double val, double inc){
        for(int i = from; i < to; ++i){
            ptr[i] = val;
            val += inc;
        }
    };
    
    opio.type = "Conv_Eigen";
    opio.inputs.push_back("x");
    opio.inputs.push_back("w");
    opio.inputs.push_back("b");
    opio.outputs.push_back("y");
    opio.param.Add("Filters", out_filters);
    opio.param.Add("Kernel", kernel_shape);
    opio.param.Add("Stride", stride_shape);
    opio.param.Add("Padding", padding);
    
    opio_grad.type = "Conv_Gradient";
    opio_grad.inputs.push_back("x");
    opio_grad.inputs.push_back("w");
    opio_grad.inputs.push_back("dy");
    opio_grad.outputs.push_back("dw");
    opio_grad.outputs.push_back("db");
    opio_grad.outputs.push_back("dx");
    opio_grad.param = opio.param;
    
    ih.AddItem<TensorBlob<CPUContext>>("x");
    ih.AddItem<TensorBlob<CPUContext>>("dy");
    x = ih.GetItem<TensorBlob<CPUContext>>("x");
    dy = ih.GetItem<TensorBlob<CPUContext>>("dy");
    x->Resize<double>({batch, 2, 6, 6});
    dy->Resize<double>(
    {
        x->Dim(0),
        out_filters,
        (x->Dim(2) + 2 * padding - kernel_shape[0]) / stride_shape[0] + 1,
        (x->Dim(3) + 2 * padding - kernel_shape[1]) / stride_shape[1] + 1
    });
    conv = CreateOperator(opio, &ih);
    conv_grad = CreateOperator(opio_grad, &ih);
    w = ih.GetItem<TensorBlob<CPUContext>>("w");
    b = ih.GetItem<TensorBlob<CPUContext>>("b");
    y = ih.GetItem<TensorBlob<CPUContext>>("y");
    dw = ih.GetItem<TensorBlob<CPUContext>>("dw");
    db = ih.GetItem<TensorBlob<CPUContext>>("db");
    dx = ih.GetItem<TensorBlob<CPUContext>>("dx");
    
    set(x->GetPtrMutable<double>(), 0, x->Size(), 1., 0.7);
    set(w->GetPtrMutable<double>(), 0, w->Size(), 1., 0.5);
    set(b->GetPtrMutable<double>(), 0, b->Size(), 1., 0.);
    set(dy->GetPtrMutable<double>(), 0, dy->Size(), 1., 0.);
    
    {
        cout<<"-- Convolution Operator Run"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        conv->Compute();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    {
        cout<<"-- Convolution Operator Gradient Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        GradientChecker<double, CPUContext> gc(0.00001);
        conv_grad->Compute();
        std::shared_ptr<TensorBlob<CPUContext>> gc_val;
        
        /*
         * check gradient w.r.t. kernel.
         */
        gc_val = gc.Run(conv, w, y, dw, 1. / batch);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. bias.
         */
        gc_val = gc.Run(conv, b, y, db, 1. / batch);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(conv, x, y, dx, 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
