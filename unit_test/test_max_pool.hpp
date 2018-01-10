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
    shared_ptr<Operator<CPUContext>> mp;
    shared_ptr<Operator<CPUContext>> mp_grad;
    ParamDef param_conv;
    vector<shared_ptr<TensorBlob<CPUContext>>> mp_inputs(1), mp_grad_inputs(3);
    vector<shared_ptr<TensorBlob<CPUContext>>> mp_outputs(2), mp_grad_outputs(1);
    const std::vector<int> kernel = {2, 2};
    const std::vector<int> stride = {2, 2};
    const int batch = 2;
    const double acceptable_gradient_check_val = 1e-7;
    
    /*
     * max pool op IO.
     */
    auto x = mp_inputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto y = mp_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    auto idx = mp_outputs[1] = make_shared<TensorBlob<CPUContext>>();
    // make x
    x->Reshape<double>({batch, 2, 6, 6});
    
    /*
     * max pool gradient op IO.
     */
    mp_grad_inputs[0] = x;
    mp_grad_inputs[1] = idx;
    auto dy = mp_grad_inputs[2] = make_shared<TensorBlob<CPUContext>>();
    auto dx = mp_grad_outputs[0] = make_shared<TensorBlob<CPUContext>>();
    // make dy
    dy->Reshape<double>(
                        {
                            batch, 2,
                            (x->Dim(2) - kernel[0]) / stride[0] + 1,
                            (x->Dim(3) - kernel[1]) / stride[1] + 1
                        });
    
    auto set = [](double *ptr, int from, int to, double val, double inc){
        for(int i = from; i < to; ++i){
            ptr[i] = val;
            val += inc;
        }
    };
    
    auto set_rand = [](double *ptr, int size, std::mt19937 &gen){
        std::uniform_int_distribution<int> uni(1, 999);
        for(int i = 0; i < size; ++i){
            ptr[i] = uni(gen);
        }
    };
    
    set(x->GetPtrMutable<double>(), 0, x->Size(), 1., 1.);
    set(dy->GetPtrMutable<double>(), 0, dy->Size(), 0.5, 0.);
    
    param_conv.Add("Kernel", kernel);
    param_conv.Add("Stride", stride);
    try{
        mp = make_shared<MaxPoolOp<double, CPUContext>>(
                                                        mp_inputs,
                                                        mp_outputs,
                                                        param_conv
                                                        );
        
        mp_grad = make_shared<MaxPoolGradientOp<double, CPUContext>>(
                                                                     mp_grad_inputs,
                                                                     mp_grad_outputs,
                                                                     param_conv
                                                                     );
    }
    catch(std::string e){
        FAIL()<<e<<std::endl;
    }
    
    {
        cout<<"-- Convolution Operator Run"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
//        mp->Compute();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    std::mt19937 rnd;
    std::cout<<-FLT_MAX<<std::endl;
    
    for(int i = 0; i < 100; ++i){
        set_rand(x->GetPtrMutable<double>(), x->Size(), rnd);
        mp->Compute();
        mp_grad->Compute();
        for(int b = 0; b < x->Dim(0); ++b){
            for(int c = 0; c < x->Dim(1); ++c){
                for(int h = 0; h < x->Dim(2); ++h){
                    for(int w = 0; w < x->Dim(3); ++w){
                        int n = b * x->Size() / x->Dim(0);
                        n += c * x->Dim(2) * x->Dim(3);
                        n += h * x->Dim(3) + w;
                        std::cout<<std::setw(4)<<x->GetPtrConst<double>()[n]<<" ";
                    }
                    std::cout<<std::endl;
                }
            }
            std::cout<<std::endl<<std::endl;
        }
        
        for(int b = 0; b < y->Dim(0); ++b){
            for(int c = 0; c < y->Dim(1); ++c){
                for(int h = 0; h < y->Dim(2); ++h){
                    for(int w = 0; w < y->Dim(3); ++w){
                        int n = b * y->Size() / y->Dim(0);
                        n += c * y->Dim(2) * y->Dim(3);
                        n += h * y->Dim(3) + w;
                        std::cout<<std::setw(4)<<y->GetPtrConst<double>()[n]<<" ";
                    }
                    std::cout<<std::endl;
                }
            }
            std::cout<<std::endl<<std::endl;
        }
        
        for(int b = 0; b < idx->Dim(0); ++b){
            for(int c = 0; c < idx->Dim(1); ++c){
                for(int h = 0; h < idx->Dim(2); ++h){
                    for(int w = 0; w < idx->Dim(3); ++w){
                        int n = b * idx->Size() / idx->Dim(0);
                        n += c * idx->Dim(2) * idx->Dim(3);
                        n += h * idx->Dim(3) + w;
                        std::cout<<std::setw(4)<<idx->GetPtrConst<int>()[n]<<" ";
                    }
                    std::cout<<std::endl;
                }
            }
            std::cout<<std::endl<<std::endl;
        }
        
        for(int b = 0; b < dx->Dim(0); ++b){
            for(int c = 0; c < dx->Dim(1); ++c){
                for(int h = 0; h < dx->Dim(2); ++h){
                    for(int w = 0; w < dx->Dim(3); ++w){
                        int n = b * dx->Size() / dx->Dim(0);
                        n += c * dx->Dim(2) * dx->Dim(3);
                        n += h * dx->Dim(3) + w;
                        std::cout<<std::setw(4)<<dx->GetPtrConst<double>()[n]<<" ";
                    }
                    std::cout<<std::endl;
                }
            }
            std::cout<<std::endl<<std::endl;
        }
        std::cout<<"-----"<<std::endl;
    }

    
    return ;
    {
        cout<<"-- Convolution Operator Gradient Check"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        GradientChecker<double, CPUContext> gc(0.00001);
        mp_grad->Compute();
        std::shared_ptr<TensorBlob<CPUContext>> gc_val;
        
        /*
         * check gradient w.r.t. kernel.
         */
        gc_val = gc.Run(mp, mp->Input(1), mp->Output(0), mp_grad->Output(0), 1. / batch);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        
        /*
         * check gradient w.r.t. input.
         */
        gc_val = gc.Run(mp, mp->Input(0), mp->Output(0), mp_grad->Output(2), 1.);
        for(int n = 0; n < gc_val->Size(); ++n){
            EXPECT_LT(gc_val->GetPtrConst<double>()[n], acceptable_gradient_check_val);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
}
