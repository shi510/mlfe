#ifndef __OPERATOR_FACTORY_HPP__
#define __OPERATOR_FACTORY_HPP__

#include <map>
#include <memory>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/fully_connected.hpp>
#include <mlfe/operators/softmax_xent_with_label.hpp>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>
#include <mlfe/operators/db_reader.hpp>
#include <mlfe/operators/cast.hpp>
#include <mlfe/operators/scale.hpp>
#include <mlfe/operators/one_hot.hpp>
#include <mlfe/operators/convolution_eigen.hpp>
#include <mlfe/operators/max_pool.hpp>
#include <mlfe/operators/relu.hpp>
#include <mlfe/operators/flatten.hpp>
#include <mlfe/operators/filler.hpp>
#include "item_holder.hpp"
#include "common.hpp"

struct OperatorInfo{
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    mlfe::ParamDef param;
};

class OperatorFactory{
public:
    OperatorFactory();
    
    OperatorCPU_Ptr GetOperator(
                                OperatorInfo op_info,
                                std::shared_ptr<ItemHolder<TensorCPU>> item_holder
                                );
    
    OperatorInfo GetGradientInfo(OperatorInfo op_info);
    
    OperatorInfo GetGradientDBReader(OperatorInfo op_info);
    
    OperatorInfo GetGradientCast(OperatorInfo op_info);
    
    OperatorInfo GetGradientScale(OperatorInfo op_info);
    
    OperatorInfo GetGradientOneHot(OperatorInfo op_info);
    
    OperatorInfo GetGradientConv(OperatorInfo op_info);
    
    OperatorInfo GetGradientMaxPool(OperatorInfo op_info);
    
    OperatorInfo GetGradientRelu(OperatorInfo op_info);
    
    OperatorInfo GetGradientFC(OperatorInfo op_info);
    
    OperatorInfo GetGradientSoftmaxXent(OperatorInfo op_info);
    
    OperatorInfo GetGradientFlatten(OperatorInfo op_info);
    
    OperatorInfo MakeOpInfo(
                            std::string type,
                            std::vector<std::string> inputs,
                            std::vector<std::string> outputs,
                            mlfe::ParamDef param
                            );
    
protected:
    std::string GI(OperatorInfo info, int n);
    
    std::string GO(OperatorInfo info, int n);
    
private:
    std::map<std::string,
                    std::function<OperatorCPU_Ptr (
                                            std::vector<TensorCPU_Ptr> &,
                                            std::vector<TensorCPU_Ptr> &,
                                            mlfe:: ParamDef &
                                            )>> ops;
    
    std::map<std::string,
                    std::function<OperatorInfo (OperatorInfo &)>> gradient_info;
};


#endif /* __OPERATOR_FACTORY_HPP__ */
