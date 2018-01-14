#include <iostream>
#include <random>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/max_pool.hpp>
#include <mlfe/core/param_def.hpp>
#include <gtest/gtest.h>
#include <mlfe/utils/gradient_checker.hpp>
#include <iomanip>
#include <random>

using namespace std;
using namespace mlfe;

TEST(MaxPoolOperatorTest, VerifyCPUResults) {
    ItemHolder ih;
    OperatorIO opio, opio_grad;
    std::shared_ptr<OperatorBase> mp, mp_grad;
    TensorBlob<CPUContext> *x, *y, *idx, *dy, *dx;
    const std::vector<int> kernel_shape = {2, 2};
    const std::vector<int> stride_shape = {2, 2};
    const int batch = 2;
    const double acceptable_gradient_check_val = 1e-7;
    auto set = [](double *ptr, int from, int to, double val, double inc){
        for(int i = from; i < to; ++i){
            ptr[i] = val;
            val += inc;
        }
    };
    
    opio.type = "MaxPool";
    opio.data_type = "double";
    opio.inputs.push_back("x");
    opio.outputs.push_back("y");
    opio.outputs.push_back("idx");
    opio.param.Add("Kernel", kernel_shape);
    opio.param.Add("Stride", stride_shape);
    
    opio_grad.type = "MaxPool_Gradient";
    opio_grad.inputs.push_back("x");
    opio_grad.inputs.push_back("idx");
    opio_grad.inputs.push_back("dy");
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
                           2,
                           (x->Dim(2) - kernel_shape[0]) / stride_shape[0] + 1,
                           (x->Dim(3) - kernel_shape[1]) / stride_shape[1] + 1
                       });
    mp = CreateOperator(opio, &ih);
    mp_grad = CreateOperator(opio_grad, &ih);
    y = ih.GetItem<TensorBlob<CPUContext>>("y");
    idx = ih.GetItem<TensorBlob<CPUContext>>("idx");
    dx = ih.GetItem<TensorBlob<CPUContext>>("dx");
    
    
    set(x->GetPtrMutable<double>(), 0, x->Size(), 1., 1.);
    set(dy->GetPtrMutable<double>(), 0, dy->Size(), 1., 0.);
    
    {
        cout<<"-- Convolution Operator Run"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        mp->Compute();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    {
        cout<<"-- Convolution Operator Gradient Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        GradientChecker<double, CPUContext> gc(0.00001);
        mp_grad->Compute();
        std::shared_ptr<TensorBlob<CPUContext>> gc_val;
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(mp, x, y, dx, 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
