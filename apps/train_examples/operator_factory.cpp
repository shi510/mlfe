#include "operator_factory.hpp"

OperatorFactory::OperatorFactory(){
    ops["cast"] = [](
                      std::vector<TensorCPU_Ptr> &inputs,
                      std::vector<TensorCPU_Ptr> &outputs,
                      mlfe::ParamDef &param
                      ) -> OperatorCPU_Ptr{
        using Op = mlfe::CastOp<mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["cast"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientCast(info);
    };
    
    ops["scale"] = [](
                          std::vector<TensorCPU_Ptr> &inputs,
                          std::vector<TensorCPU_Ptr> &outputs,
                          mlfe::ParamDef &param
                          ) -> OperatorCPU_Ptr{
        using Op = mlfe::ScaleOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["scale"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientScale(info);
    };
    
    ops["onehot"] = [](
                     std::vector<TensorCPU_Ptr> &inputs,
                     std::vector<TensorCPU_Ptr> &outputs,
                     mlfe::ParamDef &param
                     ) -> OperatorCPU_Ptr{
        using Op = mlfe::OneHotOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["onehot"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientOneHot(info);
    };
    
    ops["db_reader"] = [](
                          std::vector<TensorCPU_Ptr> &inputs,
                          std::vector<TensorCPU_Ptr> &outputs,
                          mlfe::ParamDef &param
                          ) -> OperatorCPU_Ptr{
        using Op = mlfe::DBReaderOp<unsigned char, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(outputs, param));
    };
    
    gradient_info["db_reader"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientDBReader(info);
    };
    
    ops["conv"] = [](
                     std::vector<TensorCPU_Ptr> &inputs,
                     std::vector<TensorCPU_Ptr> &outputs,
                     mlfe::ParamDef &param
                     ) -> OperatorCPU_Ptr{
        using Op = mlfe::ConvolutionWithEigenOp<float>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    ops["conv_gradient"] = [](
                            std::vector<TensorCPU_Ptr> &inputs,
                            std::vector<TensorCPU_Ptr> &outputs,
                            mlfe::ParamDef &param
                            ) -> OperatorCPU_Ptr{
        using Op = mlfe::ConvolutionGradientWithEigenOp<float>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["conv"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientConv(info);
    };
    
    ops["maxpool"] = [](
                   std::vector<TensorCPU_Ptr> &inputs,
                   std::vector<TensorCPU_Ptr> &outputs,
                   mlfe::ParamDef &param
                   ) -> OperatorCPU_Ptr{
        using Op = mlfe::MaxPoolOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    ops["maxpool_gradient"] = [](
                            std::vector<TensorCPU_Ptr> &inputs,
                            std::vector<TensorCPU_Ptr> &outputs,
                            mlfe::ParamDef &param
                            ) -> OperatorCPU_Ptr{
        using Op = mlfe::MaxPoolGradientOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["maxpool"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientMaxPool(info);
    };
    
    ops["relu"] = [](
                     std::vector<TensorCPU_Ptr> &inputs,
                     std::vector<TensorCPU_Ptr> &outputs,
                     mlfe::ParamDef &param
                     ) -> OperatorCPU_Ptr{
        using Op = mlfe::ReluOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    ops["relu_gradient"] = [](
                              std::vector<TensorCPU_Ptr> &inputs,
                              std::vector<TensorCPU_Ptr> &outputs,
                              mlfe::ParamDef &param
                              ) -> OperatorCPU_Ptr{
        using Op = mlfe::ReluGradientOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["relu"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientRelu(info);
    };
    
    ops["fc"] = [](
                   std::vector<TensorCPU_Ptr> &inputs,
                   std::vector<TensorCPU_Ptr> &outputs,
                   mlfe::ParamDef &param
                   ) -> OperatorCPU_Ptr{
        using Op = mlfe::FullyConnectedOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    ops["fc_gradient"] = [](
                            std::vector<TensorCPU_Ptr> &inputs,
                            std::vector<TensorCPU_Ptr> &outputs,
                            mlfe::ParamDef &param
                            ) -> OperatorCPU_Ptr{
        using Op = mlfe::FullyConnectedGradientOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    gradient_info["fc"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientFC(info);
    };
    
    ops["softmax_xent_loss_with_label"] = [](
                                             std::vector<TensorCPU_Ptr> &inputs,
                                             std::vector<TensorCPU_Ptr> &outputs,
                                             mlfe::ParamDef &param
                                             ) -> OperatorCPU_Ptr{
        using Op = mlfe::SoftmaxCrossEntropyWithLabelOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs));
    };
    
    ops["softmax_xent_loss_with_label_gradient"] = [](
                                                      std::vector<TensorCPU_Ptr> &inputs,
                                                      std::vector<TensorCPU_Ptr> &outputs,
                                                      mlfe::ParamDef &param
                                                      ) -> OperatorCPU_Ptr{
        using Op = mlfe::SoftmaxCrossEntropyWithLabelGradientOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs));
    };
    
    gradient_info["softmax_xent_loss_with_label"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientSoftmaxXent(info);
    };
    
    ops["flatten"] = [](
                        std::vector<TensorCPU_Ptr> &inputs,
                        std::vector<TensorCPU_Ptr> &outputs,
                        mlfe::ParamDef &param
                        ) -> OperatorCPU_Ptr{
        using Op = mlfe::FlattenOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    ops["flatten_gradient"] = [](
                                 std::vector<TensorCPU_Ptr> &inputs,
                                 std::vector<TensorCPU_Ptr> &outputs,
                                 mlfe::ParamDef &param
                                 ) -> OperatorCPU_Ptr{
        using Op = mlfe::FlattenGradientOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs));
    };
    
    gradient_info["flatten"] = [this](OperatorInfo &info) -> OperatorInfo{
        return this->GetGradientFlatten(info);
    };
    
    // Fill Ops.
    
    ops["constant_fill"] = [](
                              std::vector<TensorCPU_Ptr> &inputs,
                              std::vector<TensorCPU_Ptr> &outputs,
                              mlfe::ParamDef &param
                              ) -> OperatorCPU_Ptr{
        using Op = mlfe::ConstantFillOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
    
    ops["xavier_fill"] = [](
                        std::vector<TensorCPU_Ptr> &inputs,
                        std::vector<TensorCPU_Ptr> &outputs,
                        mlfe::ParamDef &param
                        ) -> OperatorCPU_Ptr{
        using Op = mlfe::XavierFillOp<float, mlfe::CPUContext>;
        return static_cast<OperatorCPU_Ptr>(std::make_shared<Op>(inputs, outputs, param));
    };
}

OperatorCPU_Ptr OperatorFactory::GetOperator(
                                             OperatorInfo op_info,
                                             std::shared_ptr<ItemHolder<TensorCPU>> item_holder
                                             ){
    if(ops.count(op_info.op_type) <= 0){
        throw std::string("There are no operator type. -> ") + op_info.op_type;
    }
    std::vector<TensorCPU_Ptr> inputs;
    std::vector<TensorCPU_Ptr> outputs;
    for(auto &in : op_info.inputs){
        TensorCPU_Ptr tensor;
        tensor = item_holder->GetItem(in);
        inputs.push_back(tensor);
    }
    for(auto &out : op_info.outputs){
        TensorCPU_Ptr tensor;
        tensor = item_holder->GetItem(out);
        outputs.push_back(tensor);
    }
    return ops[op_info.op_type](inputs, outputs, op_info.param);
}

OperatorInfo OperatorFactory::GetGradientInfo(OperatorInfo op_info){
    OperatorInfo op_info_grad;
    if(gradient_info.count(op_info.op_type) <= 0){
        throw std::string("There are no gradient type of ") + op_info.op_type;
    }
    return gradient_info[op_info.op_type](op_info);
}

OperatorInfo OperatorFactory::GetGradientDBReader(OperatorInfo op_info){
    return MakeOpInfo("no_gradient", {}, {}, mlfe::ParamDef());
}

OperatorInfo OperatorFactory::GetGradientScale(OperatorInfo op_info){
    return MakeOpInfo("no_gradient", {}, {}, mlfe::ParamDef());
}

OperatorInfo OperatorFactory::GetGradientCast(OperatorInfo op_info){
    return MakeOpInfo("no_gradient", {}, {}, mlfe::ParamDef());
}

OperatorInfo OperatorFactory::GetGradientOneHot(OperatorInfo op_info){
    return MakeOpInfo("no_gradient", {}, {}, mlfe::ParamDef());
}

OperatorInfo OperatorFactory::GetGradientConv(OperatorInfo op_info){
    return MakeOpInfo(
                      op_info.op_type + "_gradient",
                      {op_info.inputs[0], op_info.inputs[1], GO(op_info, 0)},
                      {GI(op_info, 1), GI(op_info, 2), GI(op_info, 0)},
                      op_info.param
                      );
}

OperatorInfo OperatorFactory::GetGradientMaxPool(OperatorInfo op_info){
    return MakeOpInfo(
                      op_info.op_type + "_gradient",
                      {op_info.inputs[0], op_info.outputs[1], GO(op_info, 0)},
                      {GI(op_info, 0)},
                      op_info.param
                      );
}

OperatorInfo OperatorFactory::GetGradientRelu(OperatorInfo op_info){
    return MakeOpInfo(
                      op_info.op_type + "_gradient",
                      {op_info.inputs[0], GO(op_info, 0)},
                      {GI(op_info, 0)},
                      op_info.param
                      );
}

OperatorInfo OperatorFactory::GetGradientFC(OperatorInfo op_info){
    return MakeOpInfo(
                      op_info.op_type + "_gradient",
                      {op_info.inputs[0], op_info.inputs[1], GO(op_info, 0)},
                      {GI(op_info, 1), GI(op_info, 2), GI(op_info, 0)},
                      op_info.param
                      );
}

OperatorInfo OperatorFactory::GetGradientSoftmaxXent(OperatorInfo op_info){
    return MakeOpInfo(
                      op_info.op_type + "_gradient",
                      {op_info.inputs[0], op_info.inputs[1], op_info.outputs[0], op_info.outputs[1]},
                      {GI(op_info, 0)},
                      op_info.param
                      );
}

OperatorInfo OperatorFactory::GetGradientFlatten(OperatorInfo op_info){
    return MakeOpInfo(
                      op_info.op_type + "_gradient",
                      {op_info.inputs[0], GO(op_info, 0)},
                      {GI(op_info, 0)},
                      op_info.param
                      );
}

OperatorInfo OperatorFactory::MakeOpInfo(
                                         std::string type,
                                         std::vector<std::string> inputs,
                                         std::vector<std::string> outputs,
                                         mlfe::ParamDef param
                                         ){
    OperatorInfo op;
    op.op_type = type;
    op.inputs = inputs;
    op.outputs = outputs;
    op.param = param;
    return op;
}

std::string OperatorFactory::GI(OperatorInfo info, int n){
    return info.inputs[n] + "_grad";
}

std::string OperatorFactory::GO(OperatorInfo info, int n){
    return info.outputs[n] + "_grad";
}
