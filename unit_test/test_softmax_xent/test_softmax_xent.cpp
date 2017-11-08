#include <iostream>
#include <cmath>
#include <chrono>
#include <mlfe/device_context/cpu_context.cpp>
#include <mlfe/operators/softmax_xent_with_label.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace mlfe;

TEST(SoftmaxXentWithLabelOperatorTest, VerifyCPUResults) {
    shared_ptr<Operator<CPUContext>> softmax_xent;
    vector<shared_ptr<TensorBlob<CPUContext>>> sm_inputs(2);
    vector<shared_ptr<TensorBlob<CPUContext>>> sm_outputs(3);
    const int batch_size = 100;
    const int x_size = 10;
    
    for(auto &in : sm_inputs){
        in = make_shared<TensorBlob<CPUContext>>();
    }
    for(auto &out : sm_outputs){
        out = make_shared<TensorBlob<CPUContext>>();
    }
    auto x = sm_inputs[0];
    auto label = sm_inputs[1];
    auto prob = sm_outputs[0];
    auto loss = sm_outputs[1];
    auto dx = sm_outputs[2];
    // make x
    x->Reshape<float>({batch_size, x_size});
    // make label
    label->ReshapeLike<float>(x);
    
    // make prob
    prob->ReshapeLike<float>(x);
    // make loss
    loss->Reshape<float>({1});
    // make dx
    dx->ReshapeLike<float>(x);
    
    auto set =[](float *p, float val, int size){ for(int i = 0; i < size; ++i){ p[i] = val; } };
    
    set(x->GetPtrMutable<float>(), 0.f, x->Size());
    set(label->GetPtrMutable<float>(), 0.f, label->Size());
    
    for(int i = 0; i < batch_size; ++i){
        for(int j = 0; j < x_size; ++j){
            if(j  != i % x_size){
                x->GetPtrMutable<float>()[i * x_size + j] = 0.5f;
            }
        }
    }
    for(int i = 0; i < batch_size; ++i){
        label->GetPtrMutable<float>()[x_size * i + i % x_size] = 1.f;
    }
    
    softmax_xent = make_shared<SoftmaxCrossEntropyWithLabel<Context::ComputePrecision::Single, CPUContext>>(sm_inputs, sm_outputs);
    
    {
        cout<<"-- Softmax Xent Operator Run (Input : 10, Output : 10, Batch : 100)"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        softmax_xent->Compute();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    {
        cout<<"-- Softmax Xent Operator Gradient Run (Input : 10, Output : 10, Batch : 100)"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        softmax_xent->ComputeGradients();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    /*
     * @brief check probability.
     */
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
            const float out_val = prob->GetPtrConst<float>()[i * x_size + j];
            float expect_unit_result;
            if(j == i % x_size){
                expect_unit_result = exp(-0.5f) / (exp(-0.5f) + 9.f);
            }
            else{
                expect_unit_result = 1.f / (exp(-0.5f) + 9.f);
            }
            EXPECT_LT(out_val, expect_unit_result + 0.01);
            EXPECT_GT(out_val, expect_unit_result - 0.01);
        }
    }

    /*
     * @brief check loss.
     */
    {
        const float expect_unit_result = -log(exp(-0.5f) / (exp(-0.5f) + 9.f));
        const float out_val = loss->GetPtrConst<float>()[0];
        EXPECT_LT(out_val, expect_unit_result + 0.01);
        EXPECT_GT(out_val, expect_unit_result - 0.01);
    }

    /*
     * @brief check dx.
     */
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
            const float out_val = dx->GetPtrConst<float>()[i * x_size + j];
            float expect_unit_result;
            if(j == i % x_size){
                expect_unit_result = (exp(-0.5f) / (exp(-0.5f) + 9.f) - 1.f) * loss->GetPtrConst<float>()[0] / batch_size;
            }
            else{
                expect_unit_result = 1.f / (exp(-0.5f) + 9.f) * loss->GetPtrConst<float>()[0] / batch_size;
            }
            EXPECT_LT(out_val, expect_unit_result + 0.01);
            EXPECT_GT(out_val, expect_unit_result - 0.01);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
