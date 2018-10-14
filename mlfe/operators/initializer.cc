#include "initializer.h"
#include "../core/op_dep.h"
#include "../core/tensor.h"
#include "../core/op_design.h"

namespace mlfe{

REGIST_OP(Constant)
    .Input("X", "float32")
    .Attr("value", "float32")
    .Finish();

REGIST_OP(Normal)
    .Input("X", "float32")
    .Attr("std", "float32")
    .Attr("clip", "bool")
    .Finish();

REGIST_OP(Xavier)
    .Input("X", "float32")
    .Attr("a", "int32")
    .Attr("b", "int32")
    .Finish();

namespace functional{

Tensor Constant(type::float64::T val, std::vector<int> shape){
    Tensor x;

    x.Reshape(shape);
    auto dep = OpDependency::Builder("Constant")
        .Input(x)
        .Attr({ "value", static_cast<type::float32::T>(val) })
        .Finish();

    x = Tensor::DependencyAdder(dep);

    return x;
}

Tensor Normal(type::float64::T std, std::vector<int> shape){
    Tensor x;

    x.Reshape(shape);
    auto dep = OpDependency::Builder("Normal")
        .Input(x)
        .Attr({ "std", static_cast<type::float32::T>(std) })
        .Attr({ "clip", static_cast<bool>(false) })
        .Finish();

    x = Tensor::DependencyAdder(dep);

    return x;
}

Tensor TruncatedNormal(type::float64::T std, std::vector<int> shape){
    Tensor x;

    x.Reshape(shape);
    auto dep = OpDependency::Builder("Normal")
        .Input(x)
        .Attr({ "std", static_cast<type::float32::T>(std) })
        .Attr({ "clip", static_cast<bool>(true) })
        .Finish();

    x = Tensor::DependencyAdder(dep);

    return x;
}

Tensor Xavier(type::int32::T a, type::int32::T b, std::vector<int> shape){
    Tensor x;

    x.Reshape(shape);
    auto dep = OpDependency::Builder("Xavier")
        .Input(x)
        .Attr({ "a", static_cast<type::int32::T>(a) })
        .Attr({ "b", static_cast<type::int32::T>(b) })
        .Finish();

    x = Tensor::DependencyAdder(dep);

    return x;
}

} // end namespace functional
} // end namespace mlfe
