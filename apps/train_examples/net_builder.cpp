#include <iostream>
#include <memory>
#include <iomanip>
#include <random>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/fully_connected.hpp>
#include <mlfe/operators/softmax_xent_with_label.hpp>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>
#include <mlfe/operators/db_reader.hpp>
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
                                     int batch_size,
                                     bool has_label
                                     ){
    std::string data = name + "_data";
    std::string label = name + "_label";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("DataBasePath", db_path);
    param.Add("DataBaseType", db_type);
    param.Add("BatchSize", batch_size);
    param.Add("HasLabel", has_label);
    param.Add("DataShape", input_dim);
    param.Add("LabelShape", std::vector<int>{batch_size, 1});
    
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

OperatorInfo NetBuilder::AddScale(std::string name, std::string x, float scaler){
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("Scale", scaler);
    
    op_info = op_fact.MakeOpInfo(
                                 "scale", {x}, {y},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto scale = std::make_pair(name, op);
    layers.push_back(scale);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add Scale Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddCast(std::string name, std::string x, std::string cast_type){
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("Cast", cast_type);
    
    op_info = op_fact.MakeOpInfo(
                                 "cast", {x}, {y},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto cast = std::make_pair(name, op);
    layers.push_back(cast);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add Cast Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddOneHot(std::string name, std::string x, int dim){
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("Dim", dim);
    
    op_info = op_fact.MakeOpInfo(
                                 "onehot", {x}, {y},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto onehot = std::make_pair(name, op);
    layers.push_back(onehot);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add One Hot Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddConv(std::string name, std::string x, int filters,
                                 std::vector<int> kernel, std::vector<int> stride, int padding){
    std::string w = name + "_w";
    std::string b = name + "_b";
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info, init_w, init_b;
    OperatorCPU_Ptr op;
    
    param.Add("Filters", filters);
    param.Add("Kernel", kernel);
    param.Add("Stride", stride);
    param.Add("Padding", padding);
    
    op_info = op_fact.MakeOpInfo(
                                 "conv", {x, w, b}, {y},
                                 param
                                 );
    init_w = op_fact.MakeOpInfo(
                                "xavier_fill", {}, {w},
                                ParamDef()
                                );
    
    init_b = op_fact.MakeOpInfo(
                                "constant_fill", {}, {b},
                                ParamDef().Add("Value", 1.f)
                                );
    
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto  fc = std::make_pair(name, op);
    layers.push_back(fc);
    init_layers.push_back(std::make_pair(name + "init_w", op_fact.GetOperator(init_w, item_holder)));
    init_layers.push_back(std::make_pair(name + "init_b", op_fact.GetOperator(init_b, item_holder)));
    op_infos.push_back(op_info);
    
    trainable_var.push_back(w);
    trainable_var.push_back(b);
    
    std::cout<<"- Add Conv Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<", "<<w<<", "<<b<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddMaxPool(std::string name, std::string x,
                                    std::vector<int> kernel, std::vector<int> stride){
    std::string y = name + "_y";
    std::string idx = name + "_idx";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("Kernel", kernel);
    param.Add("Stride", stride);
    
    op_info = op_fact.MakeOpInfo(
                                 "maxpool", {x}, {y, idx},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto  fc = std::make_pair(name, op);
    layers.push_back(fc);
    op_infos.push_back(op_info);
    
    
    std::cout<<"- Add MaxPool Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<", "<<idx<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddRelu(std::string name, std::string x, bool inplace){
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    param.Add("Inplace", inplace);
    
    op_info = op_fact.MakeOpInfo(
                                 "relu", {x}, {y},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto  relu = std::make_pair(name, op);
    layers.push_back(relu);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add Relu Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
    return op_info;
}

OperatorInfo NetBuilder::AddFC(std::string name, std::string x, int units){
    std::string w = name + "_w";
    std::string b = name + "_b";
    std::string y = name + "_y";
    ParamDef param;
    OperatorInfo op_info, init_w, init_b;
    OperatorCPU_Ptr op;
    
    param.Add("Units", units);
    
    op_info = op_fact.MakeOpInfo(
                                 "fc", {x, w, b}, {y},
                                 param
                                 );
    
    init_w = op_fact.MakeOpInfo(
                                "xavier_fill", {}, {w},
                                ParamDef()
                                );
    
    init_b = op_fact.MakeOpInfo(
                                "constant_fill", {}, {b},
                                ParamDef().Add("Value", 0)
                                );
    
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto  fc = std::make_pair(name, op);
    layers.push_back(fc);
    init_layers.push_back(std::make_pair(name + "init_w", op_fact.GetOperator(init_w, item_holder)));
    init_layers.push_back(std::make_pair(name + "init_b", op_fact.GetOperator(init_b, item_holder)));
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

OperatorInfo NetBuilder::AddFlatten(std::string name, std::string x, int axis){
    auto y = name + "_y";
    ParamDef param;
    OperatorInfo op_info;
    OperatorCPU_Ptr op;
    
    param.Add("Axis", axis);
    
    op_info = op_fact.MakeOpInfo(
                                 "flatten",
                                 {x}, {y},
                                 param
                                 );
    op = op_fact.GetOperator(op_info, item_holder);
    
    auto flatten = std::make_pair(name, op);
    layers.push_back(flatten);
    op_infos.push_back(op_info);
    
    std::cout<<"- Add Flatten Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<x<<std::endl;
    std::cout<<"    "<<"Output : "<<y<<std::endl;
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
    for(auto &op : init_layers){
        op.second->Compute();
    }
}
