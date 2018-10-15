#include "basic_arithmetics.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"
#include <iostream>

namespace mlfe{
namespace functional{

REGIST_OP(ElementwiseAdd)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x1 = odc->Input(0);
        auto &x2 = odc->Input(1);
        auto &y = odc->Output(0);
        if(x1.Size() != x2.Size()){
            throw std::string("ElementwiseAdd : "
                "the Shape of A and B is not same.");
        }
        y.Reshape(x1.Shape(), type::float32());
    })
    .Finish();

REGIST_OP_GRAD(ElementwiseAdd)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Input("dY", "float32")
    .Output("dX1", "float32")
    .Output("dX2", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &dy = odc->Input(1);
        auto &dx1 = odc->Output(0);
        auto &dx2 = odc->Output(1);
        dx1.Reshape(dy.Shape(), type::float32());
        dx2.Reshape(dy.Shape(), type::float32());
    })
    .Finish();

class ElementwiseAddGradient : public GradientHelper{
public:
    ElementwiseAddGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y,
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x1 = y.get_children()[0];
        Tensor x2 = y.get_children()[1];
        Tensor dx1, dx2;
        dep = OpDependency::Builder("ElementwiseAddGradient")
            .Input(x1)
            .Input(x2)
            .Input(dy)
            .Output(dx1)
            .Output(dx2)
            .Finish();

        gpair[x1] = dx1;
        gpair[x2] = dx2;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseAdd, ElementwiseAddGradient)

REGIST_OP(ElementwiseMul)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x1 = odc->Input(0);
        auto &x2 = odc->Input(1);
        auto &y = odc->Output(0);
        if(x1.Size() != x2.Size()){
            throw std::string("ElementwiseMul : "
                "the Shape of A and B is not same.");
        }
        y.Reshape(x1.Shape(), type::float32());
    })
    .Finish();

REGIST_OP_GRAD(ElementwiseMul)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Input("dY", "float32")
    .Output("dX1", "float32")
    .Output("dX2", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &dy = odc->Input(2);
        auto &dx1 = odc->Output(0);
        auto &dx2 = odc->Output(1);
        dx1.Reshape(dy.Shape(), type::float32());
        dx2.Reshape(dy.Shape(), type::float32());
    })
    .Finish();

class ElementwiseMulGradient : public GradientHelper{
public:
    ElementwiseMulGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y,
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x1 = y.get_children()[0];
        Tensor x2 = y.get_children()[1];
        Tensor dx1, dx2;
        dep = OpDependency::Builder("ElementwiseMulGradient")
            .Input(x1)
            .Input(x2)
            .Input(dy)
            .Output(dx1)
            .Output(dx2)
            .Finish();

        gpair[x1] = dx1;
        gpair[x2] = dx2;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseMul, ElementwiseMulGradient)

REGIST_OP(AddN)
    .Input("Xs", "float32s")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto &x1 = odc->Input(0);
        auto &c = odc->Output(0);
        int num = odc->NumInput();
        for(int n = 1; n < num; ++n){
            if(x1.Size() != odc->Input(n).Size()){
                throw std::string("AddN : "
                    "the Shape of Inputs is not same.");
            }
        }
        c.Reshape(x1.Shape(), type::float32());
    })
    .Finish();

REGIST_OP_GRAD(AddN)
    .Input("dY", "float32")
    .Output("dXs", "float32s")
    .ShapeInference([](OpDesignContext * odc){
        auto &dy = odc->Input(0);
        for(int n = 0; n < odc->NumOutput(); ++n){
            Tensor d = odc->Output(n);
            d.Reshape(dy.Shape(), type::float32());
        }
    })
    .Finish();

class AddNGradient : public GradientHelper{
public:
    AddNGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y,
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        auto odb = OpDependency::Builder("ElementwiseAddGradient")
            .Input(dy);
        for(auto &x : y.get_children()){
            Tensor d;
            odb = odb.Output(d);
            gpair[x] = d;
        }
        dep = odb.Finish();
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(AddN, AddNGradient)

template <>
Tensor Add<Tensor>(Tensor a, Tensor b){
    Tensor y;
    auto dep = OpDependency::Builder("ElementwiseAdd")
        .Input(a)
        .Input(b)
        .Output(y)
        .Finish();
    y = Tensor::DependencyAdder(dep);
    y.add_child(a);
    y.add_child(b);
    return y;
}

template <>
Tensor Sub<Tensor>(Tensor a, Tensor b){
    Tensor y;
    auto dep = OpDependency::Builder("ElementwiseSub")
        .Input(a)
        .Input(b)
        .Output(y)
        .Finish();
    y = Tensor::DependencyAdder(dep);
    y.add_child(a);
    y.add_child(b);
    return y;
}

template <>
Tensor Mul<Tensor>(Tensor a, Tensor b){
    Tensor y;
    auto dep = OpDependency::Builder("ElementwiseMul")
        .Input(a)
        .Input(b)
        .Output(y)
        .Finish();
    y = Tensor::DependencyAdder(dep);
    y.add_child(a);
    y.add_child(b);
    return y;
}

template <>
Tensor Div<Tensor>(Tensor a, Tensor b){
    Tensor y;
    auto dep = OpDependency::Builder("ElementwiseDiv")
        .Input(a)
        .Input(b)
        .Output(y)
        .Finish();
    y = Tensor::DependencyAdder(dep);
    y.add_child(a);
    y.add_child(b);
    return y;
}

template <>
Tensor Add<double>(Tensor a, double b){
    Tensor y;
    throw std::string("Tensor add op with Scalar Value is not support.");
    return y;
}

template <>
Tensor Sub<double>(Tensor a, double b){
    Tensor y;
    throw std::string("Tensor sub op with scalar value is not support.");
    return y;
}

template <>
Tensor Mul<double>(Tensor a, double b){
    Tensor y;
    throw std::string("Tensor mul op with Scalar Value is not support.");
    return y;
}

template <>
Tensor Div<double>(Tensor a, double b){
    Tensor y;
    throw std::string("Tensor div op with Scalar Value is not support.");
    return y;
}

Tensor add_n(std::vector<Tensor> xs){
    if(xs.size() >= 2){
        Tensor y;
        auto odb = OpDependency::Builder("AddN");
        for(auto &x : xs){
            odb = odb.Input(x);
            y.add_child(x);
        }
        odb = odb.Output(y);
        auto dep = odb.Finish();
        y = Tensor::DependencyAdder(dep);
        return y;
    }
    else if(xs.size() == 1){
        return xs[0];
    }
    else{
        throw std::string("functional::Add(std::vector<Tensor>) Input Size 0");
    }
}

} // end namespace functional

#define DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(OpName, Expr) \
template <>                                               \
Tensor operator##Expr<Tensor>(Tensor a, Tensor b){        \
    return functional::OpName(a, b);                      \
}

DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(Add, +)
DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(Sub, -)
DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(Mul, *)
DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(Div, /)

} // end namespace mlfe


