#include <iostream>
#include <random>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/fully_connected.hpp>
#include <gtest/gtest.h>
#include <mlfe/utils/gradient_checker.hpp>

using namespace std;
using namespace mlfe;

TEST(FullyConnectedOperatorTest, VerifyCPUResults) {
    ItemHolder ih;
    OperatorIO opio, opio_grad;
    std::shared_ptr<OperatorBase> fc, fc_grad;
    TensorBlob<CPUContext> *x, *w, *b, *y, *dy, *dw, *db, *dx;
    const int x_size = 10;
    const int batch_size = 2;
    const int out_size = 5;
    const double bias_val = 0.75f;
    const double acceptable_gradient_check_val = 1e-7;
    auto set =[](double *p, double val, int size){ for(int i = 0; i < size; ++i){ p[i] = val; } };
    
    opio.type = "FC";
    opio.inputs.push_back("x");
    opio.inputs.push_back("w");
    opio.inputs.push_back("b");
    opio.outputs.push_back("y");
    opio.param.Add("Units", out_size);
    
    opio_grad.type = "FC_Gradient";
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
    x->Resize<double>({batch_size, x_size});
    dy->Resize<double>({batch_size, out_size});
    fc = CreateOperator(opio, &ih);
    fc_grad = CreateOperator(opio_grad, &ih);
    w = ih.GetItem<TensorBlob<CPUContext>>("w");
    b = ih.GetItem<TensorBlob<CPUContext>>("b");
    y = ih.GetItem<TensorBlob<CPUContext>>("y");
    dw = ih.GetItem<TensorBlob<CPUContext>>("dw");
    db = ih.GetItem<TensorBlob<CPUContext>>("db");
    dx = ih.GetItem<TensorBlob<CPUContext>>("dx");
    
    set(x->GetPtrMutable<double>(), 1, batch_size * x_size);
    for(int i = 0; i < out_size; ++i){
        set(w->GetPtrMutable<double>() + x_size * i, static_cast<double>(i + 1) / out_size, x_size);
    }
    set(b->GetPtrMutable<double>(), bias_val, out_size);
    set(dy->GetPtrMutable<double>(), 1., dy->Size());
    
    {
        cout<<"-- FC Operator Compute Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        fc->Compute();
        /*
         * @brief check y.
         */
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                const double expect_unit_result = static_cast<double>(x_size) *
                    (static_cast<double>(j + 1) / static_cast<double>(out_size)) + bias_val;
                const double out_val = y->GetPtrConst<double>()[i * out_size + j];
                EXPECT_LT(out_val, expect_unit_result + 0.01);
                EXPECT_GT(out_val, expect_unit_result - 0.01);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    {
        cout<<"-- FC Operator Gradient Compute Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        GradientChecker<double, CPUContext> gc(0.0001);
        fc_grad->Compute();
        std::shared_ptr<TensorBlob<CPUContext>> gc_val;
        
        /*
         * check gradient w.r.t. weight.
         */
        gc_val = gc.Run(fc, w, y, dw, 1. / batch_size);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. bias.
         */
        gc_val = gc.Run(fc, b, y, db, 1. / batch_size);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n],  acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(fc, x, y, dx, 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
