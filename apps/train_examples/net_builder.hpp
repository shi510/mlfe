#ifndef __NET_BUILDER_HPP__
#define __NET_BUILDER_HPP__

#include <vector>
#include <memory>
#include <mlfe/operators/operator.hpp>
#include <mlfe/core/item_holder.hpp>

using namespace mlfe;

class NetBuilder{
public:
    NetBuilder();
    
    OperatorIO AddDBReader(
                           std::string name,
                           std::string db_path,
                           std::string db_type,
                           std::vector<int> input_dim,
                           bool has_label
                           );
    
    OperatorIO AddScale(std::string name, std::string input, float scaler);
    
    OperatorIO AddCast(std::string name, std::string input, std::string cast_type);
    
    OperatorIO AddOneHot(std::string name, std::string input, int dim);
    
    OperatorIO AddConv(std::string name, std::string input, int filters,
                         std::vector<int> kernel, std::vector<int> stride, int padding);
    
    OperatorIO AddMaxPool(std::string name, std::string input,
                            std::vector<int> kernel, std::vector<int> stride);
    
    OperatorIO AddRelu(std::string name, std::string input, bool inplace);
    
    OperatorIO AddFC(std::string name, std::string input, int units);
    
    OperatorIO AddSoftmaxXent(std::string name, std::string input, std::string label);
    
    OperatorIO AddFlatten(std::string name, std::string input, int axis);
    
    void StopGradient();
    
    void AddAllGradientOp();
    
    void Train(int iter, float lr);
    
    void Forward();
    
protected:
    void UpdateAllTrainableVariables(float lr);
    
    void InitAllTrainableVariables();
    
private:
    std::vector<std::pair<std::string, std::shared_ptr<OperatorBase>>> init_layers;
    std::vector<std::pair<std::string, std::shared_ptr<OperatorBase>>> layers;
    std::vector<std::pair<std::string, std::shared_ptr<OperatorBase>>> layers_for_test;
    std::vector<std::string> trainable_var;
    ItemHolder ih;
    int stop_gradient_pos;
};


#endif /* __NET_BUILDER_HPP__ */
