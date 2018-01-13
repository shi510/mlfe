#include <iostream>
#include <iomanip>
#include <mlfe/utils/db/simple_db.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/math/blas.hpp>
#include <opencv2/opencv.hpp>
#include "net_builder.hpp"

NetBuilder::NetBuilder(){}

OperatorIO NetBuilder::AddDBReader(
                                     std::string name,
                                     std::string db_path,
                                     std::string db_type,
                                     std::vector<int> input_dim,
                                     bool has_label
                                     ){
    OperatorIO opio;
    opio.type = "DBReader";
    opio.outputs.push_back(name + "_data");
    opio.outputs.push_back(name + "_label");
    opio.param.Add("DatabasePath", db_path);
    opio.param.Add("DatabaseType", db_type);
    opio.param.Add("HasLabel", has_label);
    opio.param.Add("DataShape", input_dim);
    opio.param.Add("LabelShape", std::vector<int>{input_dim[0], 1});
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add DBReader Operator"<<std::endl;
    std::cout<<"    "<<"Output : ";
    std::cout<<opio.outputs[0]<<", ";
    std::cout<<opio.outputs[1]<<std::endl;
    
    return opio;
}

OperatorIO NetBuilder::AddScale(std::string name, std::string x, float scaler){
    OperatorIO opio;
    opio.type = "Scale";
    opio.inputs.push_back(x);
    opio.outputs.push_back(name + "_y");
    
    opio.param.Add("Scale", scaler);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add Scale Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<opio.inputs[0]<<std::endl;
    std::cout<<"    "<<"Output : "<<opio.outputs[0]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddCast(std::string name, std::string x, std::string cast_type){
    OperatorIO opio;
    opio.type = "Cast";
    opio.inputs.push_back(x);
    opio.outputs.push_back(name + "_y");
    
    opio.param.Add("Cast", cast_type);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add Cast Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<opio.inputs[0]<<std::endl;
    std::cout<<"    "<<"Output : "<<opio.outputs[0]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddOneHot(std::string name, std::string x, int dim){
    OperatorIO opio;
    opio.type = "OneHot";
    opio.inputs.push_back(x);
    opio.outputs.push_back(name + "_y");
    opio.param.Add("Dim", dim);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add One Hot Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<opio.inputs[0]<<std::endl;
    std::cout<<"    "<<"Output : "<<opio.outputs[0]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddConv(std::string name, std::string x, int filters,
                                 std::vector<int> kernel, std::vector<int> stride, int padding){
    OperatorIO opio, init_w, init_b;
    opio.type = "Conv";
    opio.accelerator = "Eigen";
    opio.inputs.push_back(x);
    opio.inputs.push_back(name + "_w");
    opio.inputs.push_back(name + "_b");
    opio.outputs.push_back(name + "_y");
    opio.param.Add("Filters", filters);
    opio.param.Add("Kernel", kernel);
    opio.param.Add("Stride", stride);
    opio.param.Add("Padding", padding);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    init_w.type = "XavierFill";
    init_w.outputs.push_back(name + "_w");
    init_layers.push_back(std::make_pair(name + "_init_w", CreateOperator(init_w, &ih)));
    init_b.type = "ConstantFill";
    init_b.outputs.push_back(name + "_b");
    init_b.param.Add("Value", static_cast<float>(0));
    init_layers.push_back(std::make_pair(name + "_init_b", CreateOperator(init_b, &ih)));
    
    trainable_var.push_back(name + "_w");
    trainable_var.push_back(name + "_b");
    
    std::cout<<"- Add Conv Operator"<<std::endl;
    std::cout<<"    "<<"Input : ";
    std::cout<<opio.inputs[0]<<", ";
    std::cout<<opio.inputs[1]<<", ";
    std::cout<<opio.inputs[2]<<std::endl;
    std::cout<<"    "<<"Output : ";
    std::cout<<opio.outputs[0]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddMaxPool(std::string name, std::string x,
                                    std::vector<int> kernel, std::vector<int> stride){
    OperatorIO opio;
    opio.type = "MaxPool";
    opio.inputs.push_back(x);
    opio.outputs.push_back(name + "_y");
    opio.outputs.push_back(name + "_idx");
    opio.param.Add("Kernel", kernel);
    opio.param.Add("Stride", stride);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add MaxPool Operator"<<std::endl;
    std::cout<<"    "<<"Input : ";
    std::cout<<opio.inputs[0]<<std::endl;
    std::cout<<"    "<<"Output : ";
    std::cout<<opio.outputs[0]<<", ";
    std::cout<<opio.outputs[1]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddRelu(std::string name, std::string x, bool inplace){
    OperatorIO opio;
    opio.type = "Relu";
    opio.inputs.push_back(x);
    opio.outputs.push_back(name + "_y");
    opio.param.Add("Inplace", inplace);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add Relu Operator"<<std::endl;
    std::cout<<"    "<<"Input : "<<opio.inputs[0]<<std::endl;
    std::cout<<"    "<<"Output : "<<opio.outputs[0]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddFC(std::string name, std::string x, int units){
    OperatorIO opio, init_w, init_b;
    opio.type = "FC";
    opio.inputs.push_back(x);
    opio.inputs.push_back(name + "_w");
    opio.inputs.push_back(name + "_b");
    opio.outputs.push_back(name + "_y");
    opio.param.Add("Units", units);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    init_w.type = "XavierFill";
    init_w.outputs.push_back(name + "_w");
    init_layers.push_back(std::make_pair(name + "_init_w", CreateOperator(init_w, &ih)));
    init_b.type = "ConstantFill";
    init_b.outputs.push_back(name + "_b");
    init_b.param.Add("Value", static_cast<float>(0));
    init_layers.push_back(std::make_pair(name + "_init_b", CreateOperator(init_b, &ih)));
    
    trainable_var.push_back(name + "_w");
    trainable_var.push_back(name + "_b");
    
    std::cout<<"- Add FC Operator"<<std::endl;
    std::cout<<"    "<<"Input : ";
    std::cout<<opio.inputs[0]<<", ";
    std::cout<<opio.inputs[1]<<", ";
    std::cout<<opio.inputs[2]<<std::endl;
    std::cout<<"    "<<"Output : ";
    std::cout<<opio.outputs[0]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddSoftmaxXent(std::string name, std::string x, std::string label){
    OperatorIO opio;
    opio.type = "SoftmaxXentLossWithLabel";
    opio.inputs.push_back(x);
    opio.inputs.push_back(label);
    opio.outputs.push_back(name + "_prob");
    opio.outputs.push_back(name + "_loss");
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    
    std::cout<<"- Add Softmax Xent With Label Operator"<<std::endl;
    std::cout<<"    "<<"Input : ";
    std::cout<<opio.inputs[0]<<", ";
    std::cout<<opio.inputs[1]<<std::endl;;
    std::cout<<"    "<<"Output : ";
    std::cout<<opio.outputs[0]<<", ";
    std::cout<<opio.outputs[1]<<std::endl;
    return opio;
}

OperatorIO NetBuilder::AddFlatten(std::string name, std::string x, int axis){
    OperatorIO opio;
    opio.type = "Flatten";
    opio.inputs.push_back(x);
    opio.outputs.push_back(name + "_y");
    opio.param.Add("Axis", axis);
    
    layers.push_back(std::make_pair(name, CreateOperator(opio, &ih)));
    std::cout<<"- Add Flatten Operator"<<std::endl;
    std::cout<<"    "<<"Input : ";
    std::cout<<opio.inputs[0]<<std::endl;
    std::cout<<"    "<<"Output : ";
    std::cout<<opio.outputs[0]<<std::endl;
    return opio;
}

void NetBuilder::StopGradient(){
    stop_gradient_pos = layers.size() - 1;
}

void NetBuilder::AddAllGradientOp(){
    std::vector<std::pair<std::string,
    std::shared_ptr<OperatorBase>>> gradients;
    
    for(int n =  layers.size() - 1; n >= 0 ; --n){
        if(stop_gradient_pos == n){ break; }
        auto opio = layers[n].second->GetOperatorIO();
        auto name = layers[n].first + "_grad";
        auto op_grad = CreateOperatorGradient(opio, &ih);
        gradients.push_back(std::make_pair(name, op_grad));
    }
    for(auto &op_grad : gradients){
        layers.push_back(op_grad);
    }
}

void NetBuilder::Train(int iter, float lr){
    float loss_sum = 0.f;
    auto loss = ih.template GetItem<TensorBlob<CPUContext>>("softmax_xent_loss");
    InitAllTrainableVariables();
    AddAllGradientOp();
    for(int i = 1; i <= iter; ++i){
        Forward();
        UpdateAllTrainableVariables(lr);
        if (i % 100 == 0) {
            std::cout << i << " : " << loss_sum / 100.f << std::endl;
            loss_sum = 0.f;
        }
        loss_sum += loss->GetPtrConst<float>()[0];
    }
}

void NetBuilder::Forward(){
    for(int n = 0; n < layers.size(); ++n){
        layers[n].second->Compute();
    }
}

void NetBuilder::UpdateAllTrainableVariables(float lr){
    for(auto &var_name : trainable_var){
        auto var = ih.template GetItem<TensorBlob<CPUContext>>(var_name);
        auto var_grad = ih.template GetItem<TensorBlob<CPUContext>>(var_name + "_grad");
        math::axpy<float, CPUContext>(var->Size(), -lr, var_grad->GetPtrConst<float>(), var->GetPtrMutable<float>());
    }
}

void NetBuilder::InitAllTrainableVariables(){
    for(auto &op : init_layers){
        op.second->Compute();
    }
}
