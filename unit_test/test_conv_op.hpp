#include <iostream>
#include <random>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/convolution_op_eigen.hpp>
#include <mlfe/core/param_def.hpp>
#include <gtest/gtest.h>
#include <mlfe/utils/gradient_checker.hpp>

using namespace std;
using namespace mlfe;

TEST(ConvolutionOperatorTest, VerifyCPUResults) {
    shared_ptr<Operator<CPUContext>> conv;
    shared_ptr<Operator<CPUContext>> conv_grad;
    vector<shared_ptr<TensorBlob<CPUContext>>> conv_inputs(3), conv_grad_inputs(3);
    vector<shared_ptr<TensorBlob<CPUContext>>> conv_outputs(1), conv_grad_outputs(3);
    const std::vector<int> kernel_shape = {3, 3};
    const int batch = 2;
    const int out_filters = 2;
    const int stride = 1;
    const int padding = 0;
    const double acceptable_gradient_check_val = 1e-7;
    
    /*
     * convolution op IO.
     */
    auto x = conv_inputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto w = conv_inputs[1] = make_shared<TensorBlob<CPUContext>>();
    auto b = conv_inputs[2] = make_shared<TensorBlob<CPUContext>>();
    auto y = conv_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    // make x
    x->Reshape<double>({batch, 2, 6, 6});
    // make w
    w->Reshape<double>({out_filters, x->Dim(1), kernel_shape[0], kernel_shape[1]});
    // make b
    b->Reshape<double>({w->Dim(0)});
    // make y
    y->Reshape<double>(
                       {
                           x->Dim(0),
                           out_filters,
                           (x->Dim(2) + 2 * padding - kernel_shape[0]) / stride + 1,
                           (x->Dim(3) + 2 * padding - kernel_shape[1]) / stride + 1
                       }
                       );
    
    /*
     * convolution gradient op IO.
     */
    conv_grad_inputs[0] = x;
    conv_grad_inputs[1] = w;
    auto dy = conv_grad_inputs[2] = make_shared<TensorBlob<CPUContext>>();
    auto dw = conv_grad_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto db = conv_grad_outputs[1] = make_shared<TensorBlob<CPUContext>>();
    auto dx = conv_grad_outputs[2] = make_shared<TensorBlob<CPUContext>>();
    // make dy
    dy->ReshapeLike<double>(y);
    // make dw
    dw->ReshapeLike<double>(w);
    // make db
    db->ReshapeLike<double>(b);
    // make dx
    dx->ReshapeLike<double>(x);
    
    auto set = [](double *ptr, int from, int to, double val, double inc){
        for(int i = from; i < to; ++i){
            ptr[i] = val;
            val += inc;
        }
    };

    set(x->GetPtrMutable<double>(), 0, x->Size(), 1., 0.7);
    set(w->GetPtrMutable<double>(), 0, w->Size(), 1., 0.5);
    set(b->GetPtrMutable<double>(), 0, b->Size(), 1., 0.);
    set(dy->GetPtrMutable<double>(), 0, dy->Size(), 1., 0.);
    
    conv = make_shared<ConvolutionWithEigenOp<Context::ComputePrecision::Double>>(
                                                                                   conv_inputs,
                                                                                   conv_outputs,
                                                                                   ParamDef
                                                                                   ("Filters", out_filters)
                                                                                   ("Kernel", kernel_shape)
                                                                                   ("Stride", stride)
                                                                                   ("Padding", padding)
                                                                                   );
    
    conv_grad = make_shared<ConvolutionGradientWithEigenOp<Context::ComputePrecision::Double>>(
                                                                                          conv_grad_inputs,
                                                                                          conv_grad_outputs,
                                                                                          ParamDef
                                                                                          ("Filters", out_filters)
                                                                                          ("Kernel", kernel_shape)
                                                                                          ("Stride", stride)
                                                                                          ("Padding", padding)
                                                                                          );
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
        gc_val = gc.Run(conv, conv->Input(1), conv->Output(0), conv_grad->Output(0), 1. / batch);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. bias.
         */
        gc_val = gc.Run(conv, conv->Input(2), conv->Output(0), conv_grad->Output(1), 1. / batch);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(conv, conv->Input(0), conv->Output(0), conv_grad->Output(2), 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
