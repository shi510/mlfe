#ifndef __NET_BUILDER_HPP__
#define __NET_BUILDER_HPP__

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
                             int classes,
                             int batch_size,
                             float scale,
                             bool flatten,
                             bool has_label,
                             bool one_hot
                             );
    
    OperatorInfo AddFC(std::string name, std::string input, int units);
    
    OperatorInfo AddSoftmaxXent(std::string name, std::string input, std::string label);
    
    void AddAllGradientOp();
    
    void Train(int iter, float lr);
    
    void Forward();
    
protected:
    void UpdateAllTrainableVariables(float lr);
    
    void InitAllTrainableVariables();
    
private:
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
