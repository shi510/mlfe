#include <iostream>
#include <random>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/fully_connected_op.hpp>
#include <gtest/gtest.h>
#include <mlfe/utils/gradient_checker.hpp>

using namespace std;
using namespace mlfe;

TEST(FullyConnectedOperatorTest, VerifyCPUResults) {
    shared_ptr<Operator<CPUContext>> fc;
    shared_ptr<Operator<CPUContext>> fc_grad;
    vector<shared_ptr<TensorBlob<CPUContext>>> fc_inputs(3), fc_grad_inputs(3);
    vector<shared_ptr<TensorBlob<CPUContext>>> fc_outputs(1), fc_grad_outputs(3);
    const int x_size = 10;
    const int batch_size = 2;
    const int out_size = 5;
    const double bias_val = 0.75f;
    const double acceptable_gradient_check_val = 1e-7;
    
    /*
     * fc op IO.
     */
    auto x = fc_inputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto w = fc_inputs[1] = make_shared<TensorBlob<CPUContext>>();
    auto b = fc_inputs[2] = make_shared<TensorBlob<CPUContext>>();
    auto y = fc_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    // make x
    x->Reshape<double>({batch_size, x_size});
    // make w
    w->Reshape<double>({out_size, x_size});
    // make b
    b->Reshape<double>({out_size});
    // make y
    y->Reshape<double>({batch_size, out_size});
    
    /*
     * fc gradient op IO.
     */
    fc_grad_inputs[0] = x;
    fc_grad_inputs[1] = w;
    auto dy = fc_grad_inputs[2] = make_shared<TensorBlob<CPUContext>>();
    auto dw = fc_grad_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto db = fc_grad_outputs[1] = make_shared<TensorBlob<CPUContext>>();
    auto dx = fc_grad_outputs[2] = make_shared<TensorBlob<CPUContext>>();
    // make dy
    dy->ReshapeLike<double>(y);
    // make dw
    dw->ReshapeLike<double>(w);
    // make db
    db->ReshapeLike<double>(b);
    // make dx
    dx->ReshapeLike<double>(x);
    
    auto set =[](double *p, double val, int size){ for(int i = 0; i < size; ++i){ p[i] = val; } };
    
    set(x->GetPtrMutable<double>(), 1, batch_size * x_size);
    for(int i = 0; i < out_size; ++i){
        set(w->GetPtrMutable<double>() + x_size * i, static_cast<double>(i + 1) / out_size, x_size);
    }
    set(b->GetPtrMutable<double>(), bias_val, out_size);
    set(dy->GetPtrMutable<double>(), 1., dy->Size());
    
    fc = make_shared<FullyConnectedOp<Context::ComputePrecision::Double, CPUContext>>(fc_inputs, fc_outputs);
    fc_grad = make_shared<FullyConnectedGradientOp<Context::ComputePrecision::Double, CPUContext>>(fc_grad_inputs, fc_grad_outputs);
    
    {
        cout<<"-- FC Operator Compute Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        fc->Compute();
        /*
         * @brief check y.
         */
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                const double expect_unit_result = static_cast<double>(x_size) * (static_cast<double>(j + 1) / static_cast<double>(out_size)) + bias_val;
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
        gc_val = gc.Run(fc, fc->Input(1), fc->Output(0), fc_grad->Output(0), 1. / batch_size);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. bias.
         */
        gc_val = gc.Run(fc, fc->Input(2), fc->Output(0), fc_grad->Output(1), 1. / batch_size);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n],  acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(fc, fc->Input(0), fc->Output(0), fc_grad->Output(2), 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
