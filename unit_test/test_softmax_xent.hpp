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
    ItemHolder ih;
    OperatorIO opio;
    std::shared_ptr<OperatorBase> softmax_xent, softmax_xent_grad;
    TensorBlob<CPUContext> *x, *label, *prob, *loss, *dx;
    vector<shared_ptr<TensorBlob<CPUContext>>> sm_inputs(2), sm_grad_inputs(4);
    vector<shared_ptr<TensorBlob<CPUContext>>> sm_outputs(2), sm_grad_outputs(1);
    const int batch_size = 2;
    const int x_size = 5;
    const double acceptable_gradient_check_val = 1e-7;
    auto set = [](double *ptr, int from, int to, double val, double inc){
        for(int i = from; i < to; ++i){
            ptr[i] = val;
            val += inc;
        }
    };
    
    opio.type = "SoftmaxXentLossWithLabel";
    opio.inputs.push_back("x");
    opio.inputs.push_back("label");
    opio.outputs.push_back("prob");
    opio.outputs.push_back("loss");
    
    ih.AddItem<TensorBlob<CPUContext>>("x");
    ih.AddItem<TensorBlob<CPUContext>>("label");
    x = ih.GetItem<TensorBlob<CPUContext>>("x");
    label = ih.GetItem<TensorBlob<CPUContext>>("label");
    x->Resize<double>({batch_size, x_size});
    label->Resize<double>({batch_size, x_size});
    softmax_xent = CreateOperator(opio, &ih);
    softmax_xent_grad = CreateOperatorGradient(opio, &ih);
    prob = ih.GetItem<TensorBlob<CPUContext>>("prob");
    loss = ih.GetItem<TensorBlob<CPUContext>>("loss");
    dx = ih.GetItem<TensorBlob<CPUContext>>("x_grad");
    
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
                        x,
                        loss,
                        dx,
                        loss->template GetPtrConst<double>()[0]
                        );
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
