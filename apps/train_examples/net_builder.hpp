#ifndef __NET_BUILDER_HPP__
#define __NET_BUILDER_HPP__

#include <random>
#include <mlfe/utils/db/simple_db.hpp>
#include <mlfe/operators/operator.hpp>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include "operator_factory.hpp"

using namespace mlfe;

class NetBuilder{
public:
    NetBuilder();
    
    OperatorInfo AddDBReader(
                             std::string name,
                             std::string db_path,
                             std::string db_type,
                             std::vector<int> input_dim,
                             int batch_size,
                             bool has_label
                             );
    
    OperatorInfo AddScale(std::string name, std::string input, float scaler);
    
    OperatorInfo AddCast(std::string name, std::string input, std::string cast_type);
    
    OperatorInfo AddOneHot(std::string name, std::string input, int dim);
    
    OperatorInfo AddConv(std::string name, std::string input, int filters,
                         std::vector<int> kernel, std::vector<int> stride, int padding);
    
    OperatorInfo AddMaxPool(std::string name, std::string input,
                            std::vector<int> kernel, std::vector<int> stride);
    
    OperatorInfo AddRelu(std::string name, std::string input, bool inplace);
    
    OperatorInfo AddFC(std::string name, std::string input, int units);
    
    OperatorInfo AddSoftmaxXent(std::string name, std::string input, std::string label);
    
    OperatorInfo AddFlatten(std::string name, std::string input, int axis);
    
    void AddAllGradientOp();
    
    void Train(int iter, float lr);
    
    void Forward();
    
protected:
    void UpdateAllTrainableVariables(float lr);
    
    void InitAllTrainableVariables();
    
private:
    std::vector<std::pair<std::string, OperatorCPU_Ptr>> init_layers;
    std::vector<std::pair<std::string, OperatorCPU_Ptr>> layers;
    std::vector<std::pair<std::string, OperatorCPU_Ptr>> layers_for_test;
    WorkSpace<std::function<OperatorCPU_Ptr (
                                      std::vector<TensorCPU_Ptr>,
                                      std::vector<TensorCPU_Ptr>,
                                      ParamDef
                                      )>> ops;
    
    std::vector<std::string> trainable_var;
    std::vector<OperatorInfo> op_infos;
    std::shared_ptr<ItemHolder<TensorCPU>> item_holder;
    OperatorFactory op_fact;
};


#endif /* __NET_BUILDER_HPP__ */
