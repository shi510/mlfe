#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "basic_arithmetics.h"
#include "matmul.h"
#include "initializer.h"
#include <algorithm>

namespace mlfe{
namespace functional{

REGIST_OP(ElementwiseAdd)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto x2 = odc->Input(1);
        auto y = odc->Output(0);
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
        auto dy = odc->Input(1);
        auto dx1 = odc->Output(0);
        auto dx2 = odc->Output(1);
        dx1.Reshape(dy.Shape(), type::float32());
        dx2.Reshape(dy.Shape(), type::float32());
    })
    .Finish();

class ElementwiseAddGradient : public GradientHelper{
public:
    ElementwiseAddGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x1 = y.get_children()[0];
        Tensor x2 = y.get_children()[1];
        Tensor one = functional::constant(1, y.Shape());

        in_grads.push_back(dy);
        in_grads.push_back(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseAdd, ElementwiseAddGradient)

REGIST_OP(ElementwiseMul)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto x2 = odc->Input(1);
        auto y = odc->Output(0);
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
        auto dy = odc->Input(2);
        auto dx1 = odc->Output(0);
        auto dx2 = odc->Output(1);
        dx1.Reshape(dy.Shape(), type::float32());
        dx2.Reshape(dy.Shape(), type::float32());
    })
    .Finish();

class ElementwiseMulGradient : public GradientHelper{
public:
    ElementwiseMulGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x1 = y.get_children()[0];
        Tensor x2 = y.get_children()[1];
        Tensor dx1 = functional::mul(x2, dy);
        Tensor dx2 = functional::mul(x1, dy);
        in_grads.push_back(dx1);
        in_grads.push_back(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseMul, ElementwiseMulGradient)

REGIST_OP(AddN)
    .Input("Xs", "float32s")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto c = odc->Output(0);
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
        auto dy = odc->Input(0);
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

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x1 = y.get_children()[0];
        Tensor x2 = y.get_children()[1];
        Tensor dx1 = functional::mul(x2, dy);
        Tensor dx2 = functional::mul(x1, dy);
        in_grads.push_back(dx1);
        in_grads.push_back(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(AddN, AddNGradient)

class MatrixVectorAddGradient : public GradientHelper{
public:
    MatrixVectorAddGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor mat = y.get_children()[0];
        Tensor vec = y.get_children()[1];
        Tensor one = functional::constant(1, {y.Shape()[0], 1});
        Tensor dvec = functional::matmul(dy, one, true);
        one.eval();
        in_grads.push_back(dy);
        in_grads.push_back(dvec);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(MatrixVectorAdd, MatrixVectorAddGradient)

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


Tensor add(Tensor x1, Tensor x2){
    Tensor y;
    auto x1_shape = x1.Shape();
    auto x2_shape = x2.Shape();
    auto max_dim = std::max(x1_shape.size(), x2_shape.size());
    auto min_dim = std::min(x1_shape.size(), x2_shape.size());
    std::vector<int> max_shape, min_shape;
    if(max_dim == x1_shape.size()){
        max_shape = x1_shape;
        min_shape = x2_shape;
    }
    else{
        max_shape = x2_shape;
        min_shape = x1_shape;
    }
    y = functional::create_variable(max_shape);
    y.add_child(x1);
    y.add_child(x2);
    if(max_dim == 4 && min_dim == 1){
        if(max_shape[1] == min_shape[0]){
            OpAlgoContext cxt("BatchedMatrixVectorAdd");
            Tensor::AssignOpFunctor(y, cxt);
        }
        else{
            std::string err = "Can not add with ";
            err += std::to_string(max_dim) + "d";
            err += " and ";
            err += std::to_string(min_dim) + "d";
            throw err;
        }
    }
    else if(max_dim == 2 && min_dim == 1){
        if(max_shape[1] == min_shape[0]){
            OpAlgoContext cxt("MatrixVectorAdd");
            Tensor::AssignOpFunctor(y, cxt);
        }
        else{
            std::string err = "Can not add with ";
            err += std::to_string(max_dim) + "d";
            err += " and ";
            err += std::to_string(min_dim) + "d";
            throw err;
        }
    }
    else if(max_dim == min_dim){
        OpAlgoContext cxt("ElementwiseAdd");
        Tensor::AssignOpFunctor(y, cxt);
    }

    return y;
}

Tensor mul(Tensor x1, Tensor x2){
    Tensor y = functional::create_variable(x1.Shape());
    OpAlgoContext cxt("ElementwiseMul");
    y.add_child(x1);
    y.add_child(x2);
    Tensor::AssignOpFunctor(y, cxt);

    return y;
}

Tensor add_n(std::vector<Tensor> xs){
    if(xs.size() >= 2){
        Tensor y = functional::create_variable(xs[0].Shape());
        OpAlgoContext cxt("AddN");
        for(auto &x : xs){
            y.add_child(x);
        }
        Tensor::AssignOpFunctor(y, cxt);
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
Tensor operator Expr<Tensor>(Tensor a, Tensor b){         \
    return functional::OpName(a, b);                      \
}

DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(add, +)
//DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(sub, -)
DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(mul, *)
//DEFINE_BASIC_ARITHMETIC_TENSOR_EXPR(div, /)

} // end namespace mlfe
