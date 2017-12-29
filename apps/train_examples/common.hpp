#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <map>
#include <memory>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/operators/operator.hpp>

using TensorCPU = mlfe::TensorBlob<mlfe::CPUContext>;
using TensorCPU_Ptr = std::shared_ptr<mlfe::TensorBlob<mlfe::CPUContext>>;
using OperatorCPU = mlfe::Operator<mlfe::CPUContext>;
using OperatorCPU_Ptr = std::shared_ptr<OperatorCPU>;
template <class T>
using WorkSpace = std::map<std::string, T>;

#endif /* __COMMON_HPP__ */
