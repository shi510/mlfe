#include <iostream>
#include <cmath>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/softmax_xent_with_label.hpp>
#include <gtest/gtest.h>
#include <mlfe/utils/gradient_checker.hpp>

using namespace std;
using namespace mlfe;

TEST(SoftmaxXentWithLabelOperatorTest, VerifyCPUResults) {
    shared_ptr<Operator<CPUContext>> softmax_xent;
    shared_ptr<Operator<CPUContext>> softmax_xent_grad;
    vector<shared_ptr<TensorBlob<CPUContext>>> sm_inputs(2), sm_grad_inputs(4);
    vector<shared_ptr<TensorBlob<CPUContext>>> sm_outputs(2), sm_grad_outputs(1);
    const int batch_size = 2;
    const int x_size = 5;
    const double acceptable_gradient_check_val = 1e-7;
    
    /*
     * softmax xent with loss op IO.
     */
    auto x = sm_inputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto label = sm_inputs[1] = make_shared<TensorBlob<CPUContext>>();
    auto prob = sm_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto loss = sm_outputs[1] = make_shared<TensorBlob<CPUContext>>();
    // make x
    x->Reshape<double>({batch_size, x_size});
    // make label
    label->ReshapeLike<double>(x);
    // make prob
    prob->ReshapeLike<double>(x);
    // make loss
    loss->Reshape<double>({1});
    
    /*
     * softmax xent with loss gradient op IO.
     */
    sm_grad_inputs[0] = x;
    sm_grad_inputs[1] = label;
    sm_grad_inputs[2] = prob;
    sm_grad_inputs[3] = loss;
    auto dx = sm_grad_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    // make dx
    dx->ReshapeLike<double>(x);
    
    auto set = [](double *ptr, int from, int to, double val, double inc){
        for(int i = from; i < to; ++i){
            ptr[i] = val;
            val += inc;
        }
    };
    
    set(x->GetPtrMutable<double>(), 0, x->Size(), 1., 0.5);
    set(label->GetPtrMutable<double>(), 0, label->Size(), 0., 0.);
    
    for(int i = 0; i < batch_size; ++i){
        for(int j = 0; j < x_size; ++j){
            if(j  != i % x_size){
                x->GetPtrMutable<double>()[i * x_size + j] = 0.5;
            }
        }
    }
    for(int i = 0; i < batch_size; ++i){
        label->GetPtrMutable<double>()[x_size * i + i % x_size] = 1.;
    }
    
    softmax_xent = make_shared<
    SoftmaxCrossEntropyWithLabelOp<Context::ComputePrecision::Double,CPUContext>
    >(sm_inputs, sm_outputs);
    
    softmax_xent_grad = make_shared<
    SoftmaxCrossEntropyWithLabelGradientOp<Context::ComputePrecision::Double, CPUContext>
    >(sm_grad_inputs, sm_grad_outputs);
    
    {
        cout<<"-- Softmax Xent Operator Run"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        softmax_xent->Compute();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    {
        cout<<"-- Softmax Xent Operator Gradient Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        GradientChecker<double, CPUContext> gc(0.0000001);
        std::shared_ptr<TensorBlob<CPUContext>> gc_val;
        softmax_xent_grad->Compute();
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(
                        softmax_xent,
                        softmax_xent->Input(0),
                        softmax_xent->Output(1),
                        softmax_xent_grad->Output(0),
                        softmax_xent->Output(1)->template GetPtrConst<double>()[0]
                        );
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
