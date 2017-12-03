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
    vector<shared_ptr<TensorBlob<CPUContext>>> fc_inputs(4);
    vector<shared_ptr<TensorBlob<CPUContext>>> fc_outputs(4);
    const int x_size = 100;
    const int batch_size = 3;
    const int out_size = 10;
    const double bias_val = 0.75f;
    const double acceptable_gradient_check_val = 1e-7;
    
    for(auto &in : fc_inputs){
        in = make_shared<TensorBlob<CPUContext>>();
    }
    for(auto &out : fc_outputs){
        out = make_shared<TensorBlob<CPUContext>>();
    }
    auto x = fc_inputs[0];
    auto w = fc_inputs[1];
    auto b = fc_inputs[2];
    auto dy = fc_inputs[3];
    auto y = fc_outputs[0];
    auto dw = fc_outputs[1];
    auto db = fc_outputs[2];
    auto dx = fc_outputs[3];
    // make x
    x->Reshape<double>({batch_size, x_size});
    // make w
    w->Reshape<double>({out_size, x_size});
    // make b
    b->Reshape<double>({out_size});
    // make dy
    dy->Reshape<double>({batch_size, out_size});
    
    // make y
    y->ReshapeLike<double>(dy);
    // make dw
    dw->ReshapeLike<double>(w);
    // make db
    db->ReshapeLike<double>(b);
    // make dx
    dx->ReshapeLike<double>(x);
    
    auto set =[](double *p, double val, int size){ for(int i = 0; i < size; ++i){ p[i] = val; } };
    
    set(x->GetPtrMutable<double>(), 1, batch_size * x_size);
    for(int i = 0; i < out_size; ++i){
        set(w->GetPtrMutable<double>() + x_size * i, static_cast<double>(i + 1) / 10., x_size);
    }
    set(b->GetPtrMutable<double>(), bias_val, out_size);
    set(dy->GetPtrMutable<double>(), 1., dy->Size());
    
    fc = make_shared<FullyConnectedOp<Context::ComputePrecision::Double, CPUContext>>(fc_inputs, fc_outputs);
    
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
        GradientChecker<double, CPUContext> gc(0.001);
        fc->ComputeGradients();
        std::shared_ptr<TensorBlob<CPUContext>> gc_val;
        
        /*
         * check gradient w.r.t. weight.
         */
        gc_val = gc.Run(fc, fc->Input(1), fc->Output(0), fc->Output(1), batch_size);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. bias.
         */
        gc_val = gc.Run(fc, fc->Input(2), fc->Output(0), fc->Output(2), batch_size);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n],  acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(fc, fc->Input(0), fc->Output(0), fc->Output(3), 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
