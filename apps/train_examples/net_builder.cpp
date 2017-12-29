#include <iostream>
#include <memory>
#include <iomanip>
#include <random>
#include <mlfe/core/net_sequence.hpp>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/fully_connected_op.hpp>
#include <mlfe/operators/softmax_xent_with_label.hpp>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>
#include <mlfe/operators/db_reader_op.hpp>
#include <opencv2/opencv.hpp>
#include "net_builder.hpp"

NetBuilder::NetBuilder(){
    item_holder = std::make_shared<ItemHolder<TensorCPU>>();
}

OperatorInfo NetBuilder::AddDBReader(
                                     std::string name,
                                     std::string db_path,
                                     std::string db_type,
                                     std::vector<int> input_dim,
                                     int classes,
                                     int batch_size,
                                     float scale,
                                     bool flatten,
                                     bool has_label,
                                     bool one_hot
                                     ){
    std::string data = name + "_data";
    std::string label = name + "_label";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("DataBasePath", db_path);
    param.Add("DataBaseType", db_type);
    param.Add("BatchSize", batch_size);
    param.Add("Flatten", flatten);
    param.Add("HasLabel", has_label);
    param.Add("OneHotLabel", one_hot);
    param.Add("Classes", classes);
    param.Add("Scale", scale);
    param.Add("DataShape", input_dim);
    param.Add("LabelShape", std::vector<int>{batch_size, one_hot ? classes : 1});
    
    op_info = op_fact.MakeOpInfo(
                                 "db_reader", {},
                                 {data, label},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto fc = std::make_pair(name, op);
    layers.push_back(fc);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add DBReader Operator"<<std::endl;
    std::cout<<"    "<<"Output : "<<data<<", "<<label<<std::endl;
    
    return op_info;
}

OperatorInfo NetBuilder::AddFC(std::string name, std::string x, int units){
    std::string w = name + "_w";
    std::string b = name + "_b";
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("Units", units);
    
    op_info = op_fact.MakeOpInfo(
                                 "fc", {x, w, b}, {y},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto  fc = std::make_pair(name, op);
    layers.push_back(fc);
    op_infos.push_back(op_info);
    
    trainable_var.push_back(w);
    trainable_var.push_back(b);
    
    std::cout<<"- Add FC Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<", "<<w<<", "<<b<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddSoftmaxXent(std::string name, std::string x, std::string label){
    auto prob = name + "_prob";
    auto loss = name + "_loss";
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    op_info = op_fact.MakeOpInfo(
                                 "softmax_xent_loss_with_label",
                                 {x, label}, {prob, loss},
                                 ParamDef()
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto xent = std::make_pair(name, op);
    layers.push_back(xent);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add Softmax Xent With Label Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<", "<<label<<std::endl;
    std::cout<<"    "<<"Output : "<<prob<<", "<<loss<<std::endl;
    return op_info;
}

void NetBuilder::AddAllGradientOp(){
    std::vector<std::pair<std::string, OperatorCPU_Ptr>> layers_gradient;
    for(int n = layers.size() - 1; n >= 0; --n){
        std::vector<TensorCPU_Ptr> inputs;
        std::vector<TensorCPU_Ptr> outputs;
        auto op_grad_info = op_fact.GetGradientInfo(op_infos[n]);
        if(op_grad_info.op_type.compare("no_gradient") != 0){
            auto op_grad = op_fact.GetOperator(op_grad_info, item_holder);
            auto gradient = std::make_pair(layers[n].first + "_gradient", op_grad);
            layers.push_back(gradient);
            
            std::cout<<"- Add "<<op_grad_info.op_type<<", "<<gradient.first<<std::endl;
            std::cout<<"    "<<"Input : ";
            for(auto &in : op_grad_info.inputs){
                std::cout<<in<<", ";
            }
            std::cout<<std::endl;
            std::cout<<"    "<<"Output : ";
            for(auto &out : op_grad_info.outputs){
                std::cout<<out<<", ";
            }
            std::cout<<std::endl;
        }
    }
}

void NetBuilder::Train(int iter, float lr){
    InitAllTrainableVariables();
    AddAllGradientOp();
    for(int i = 0; i < iter; ++i){
        auto loss = item_holder->GetItem("softmax_xent_loss");
        Forward();
        UpdateAllTrainableVariables(lr);
        std::cout<<i<<" : "<<loss->GetPtrConst<float>()[0]<<", "<<std::endl;
    }
}

void NetBuilder::Forward(){
    for(int n = 0; n < layers.size(); ++n){
        layers[n].second->Compute();
    }
}

void NetBuilder::UpdateAllTrainableVariables(float lr){
    for(auto &var_name : trainable_var){
        auto var = item_holder->GetItem(var_name);
        auto var_grad = item_holder->GetItem(var_name + "_grad");
        math::axpy<float, CPUContext>(var->Size(), -lr, var_grad->GetPtrConst<float>(), var->GetPtrMutable<float>());
    }
}

void NetBuilder::InitAllTrainableVariables(){
    for(auto &var_name : trainable_var){
        TensorCPU_Ptr var = item_holder->GetItem(var_name);
        float *ptr = var->GetPtrMutable<float>();
        
        for(int n = 0; n < var->Size(); ++n){
            ptr[n] = 0.f;
        }
    }
}
