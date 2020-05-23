#pragma once
#include "mlfe/utils/onnx/proto/onnx.proto3.pb.h"
#include "mlfe/core/graph.h"
#include "mlfe/core/registry.h"
#include "mlfe/core/tensor.h"
#include <string>
#include <memory>
#include <map>

namespace mlfe {
namespace onnx {

struct export_converter{
    virtual void convert(node from, ::onnx::NodeProto* to, ::onnx::GraphProto* graph) = 0;
};

class export_registry : public registry<std::shared_ptr<export_converter>>{};

template <typename T>
struct export_registerer
    : public registerer<export_registerer<T>, export_registry>{
    static std::shared_ptr<export_converter> create(){
        auto cvt = std::make_shared<T>();
        return cvt;
    }
};

template <typename Derived>
class export_impl
    : public export_converter, public export_registerer<Derived> {};

//
// end of export registry.
//

struct import_converter{
    virtual void convert(const ::onnx::NodeProto& from,
        std::map<std::string, Tensor>& inputs) = 0;
};

class import_registry
    : public registry<std::shared_ptr<import_converter>> {};

template <typename T>
struct import_registerer
    : public registerer<import_registerer<T>, import_registry>{
    static std::shared_ptr<import_converter> create(){
        auto cvt = std::make_shared<T>();
        return cvt;
    }
};

template <typename Derived>
class import_impl
    : public import_converter, public import_registerer<Derived> {};

//
// end of import registry.
//

} // end namesapce onnx
} // end namespace mlfe