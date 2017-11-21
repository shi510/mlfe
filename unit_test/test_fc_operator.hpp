#include <iostream>
#include <random>
#include <chrono>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/fully_connected_op.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace mlfe;

TEST(FullyConnectedOperatorTest, VerifyCPUResults) {
    shared_ptr<Operator<CPUContext>> fc;
    vector<shared_ptr<TensorBlob<CPUContext>>> fc_inputs(4);
    vector<shared_ptr<TensorBlob<CPUContext>>> fc_outputs(4);
    const int x_size = 784;
    const int batch_size = 10;
    const int out_size = 10;
    const float bias_val = 0.75f;
    
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
    x->Reshape<float>({batch_size, x_size});
    // make w
    w->Reshape<float>({out_size, x_size});
    // make b
    b->Reshape<float>({out_size});
    // make dy
    dy->Reshape<float>({batch_size, out_size});
    
    // make y
    y->ReshapeLike<float>(dy);
    // make dw
    dw->ReshapeLike<float>(w);
    // make db
    db->ReshapeLike<float>(b);
    // make dx
    dx->ReshapeLike<float>(x);
    
    auto set =[](float *p, float val, int size){ for(int i = 0; i < size; ++i){ p[i] = val; } };
    
    set(x->GetPtrMutable<float>(), 1, batch_size * x_size);
    for(int i = 0; i < out_size; ++i){
        set(w->GetPtrMutable<float>() + x_size * i, static_cast<float>(i + 1) / 10.f, x_size);
    }
    set(b->GetPtrMutable<float>(), bias_val, out_size);
    set(dy->GetPtrMutable<float>(), 1.f, dy->Size());
    
    fc = make_shared<FullyConnectedOp<Context::ComputePrecision::Single, CPUContext>>(fc_inputs, fc_outputs);
    
    {
        cout<<"-- FC Operator Run (Input : 784, Output : 10, Batch : 10)"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        fc->Compute();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    {
        cout<<"-- FC Operator Gradient Run (Input : 784, Output : 10, Batch : 10)"<<endl;
        auto begin = std::chrono::high_resolution_clock::now();
        fc->ComputeGradients();
        auto end = std::chrono::high_resolution_clock::now();
        cout<<"-- Total Calcaulation time : ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    
    /*
     * @brief check y.
     */
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_size; ++j) {
            const float expect_unit_result = static_cast<float>(x_size) * (static_cast<float>(j + 1) / static_cast<float>(out_size)) + bias_val;
            const float out_val = y->GetPtrConst<float>()[i * out_size + j];
            EXPECT_LT(out_val, expect_unit_result + 0.01);
            EXPECT_GT(out_val, expect_unit_result - 0.01);
        }
    }
    /*
     * @brief check dw.
     */
    for (int i = 0; i < out_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
            const float expect_unit_result = 1.f;
            const float out_val = dw->GetPtrConst<float>()[i * x_size + j];
            EXPECT_LT(out_val, expect_unit_result + 0.01);
            EXPECT_GT(out_val, expect_unit_result - 0.01);
        }
    }
    
    /*
     * @brief check db.
     */
    for (int i = 0; i < out_size; ++i) {
        const float expect_unit_result = 1.f;
        const float out_val = db->GetPtrConst<float>()[i];
        EXPECT_LT(out_val, expect_unit_result + 0.01);
        EXPECT_GT(out_val, expect_unit_result - 0.01);
    }
    
    /*
     * @brief check dx.
     */
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
            const float expect_unit_result = 11.f * 5.f / static_cast<float>(out_size);
            const float out_val = dx->GetPtrConst<float>()[i * x_size + j];
            EXPECT_LT(out_val, expect_unit_result + 0.01);
            EXPECT_GT(out_val, expect_unit_result - 0.01);
        }
    }
}
