#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"
#include "mlfe/math/transform.h"
#include "basic_arithmetics.h"
#include "matmul.h"
#include "initializer.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace mlfe{
namespace functional{

class NegativeGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        auto x = ctx_y.get_input(0);
        auto dx = functional::negative(dy);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Negative, NegativeGradient)

class ElementwiseAddGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        y.get_context().get_input(0).set_backprop_node(dy.get_node());
        y.get_context().get_input(0).set_gradient(dy);
        y.get_context().get_input(1).set_backprop_node(dy.get_node());
        y.get_context().get_input(1).set_gradient(dy);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseAdd, ElementwiseAddGradient)

class ElementwiseSubGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto x1 = y.get_context().get_input(0);
        auto x2 = y.get_context().get_input(1);
        auto dx2 = functional::negative(dy);
        x1.set_backprop_node(dy.get_node());
        x1.set_gradient(dy);
        x2.set_backprop_node(dx2.get_node());
        x2.set_gradient(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseSub, ElementwiseSubGradient)

class ElementwiseMulGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x1 = y.get_context().get_input(0);
        Tensor x2 = y.get_context().get_input(1);
        Tensor dx1 = functional::mul(x2, dy);
        Tensor dx2 = functional::mul(x1, dy);
        x1.set_backprop_node(dx1.get_node());
        x1.set_gradient(dx1);
        x2.set_backprop_node(dx2.get_node());
        x2.set_gradient(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseMul, ElementwiseMulGradient)

class ElementwiseDivGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x1 = y.get_context().get_input(0);
        Tensor x2 = y.get_context().get_input(1);
        auto one = functional::constant(1, x2.shape());
        auto dx1 = functional::div(one, x2);
        auto dx2 = functional::negative(functional::div(y, x2));
        x1.set_backprop_node(dx1.get_node());
        x1.set_gradient(dx1);
        x2.set_backprop_node(dx2.get_node());
        x2.set_gradient(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ElementwiseDiv, ElementwiseDivGradient)

class AddWithBroadcastGradient : public GradientHelper {
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override {
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        auto x1 = ctx_y.get_input(0);
        auto x2 = ctx_y.get_input(1);
        Tensor dx1 = functional::create_variable(x1.shape());
        Tensor dx2 = functional::create_variable(x2.shape());
        OpAlgoContext ctx_x_grad("AddWithBroadcastGradient");
        ctx_x_grad.add_input(dy);
        ctx_x_grad.add_input(x1);
        ctx_x_grad.add_input(x2);
        ctx_x_grad.add_input(y);
        ctx_x_grad.add_output(dx1);
        ctx_x_grad.add_output(dx2);
        dx1.set_context(ctx_x_grad);
        dx2.set_context(ctx_x_grad);
        x1.set_backprop_node(dx1.get_node());
        x1.set_gradient(dx1);
        x2.set_backprop_node(dx2.get_node());
        x2.set_gradient(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(AddWithBroadcast, AddWithBroadcastGradient)

class AddNGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        for(int n = 0; n < ctx_y.num_inputs(); ++n){
            ctx_y.get_input(n).set_backprop_node(dy.get_node());
            ctx_y.get_input(n).set_gradient(dy);
        }
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(AddN, AddNGradient)

class MatrixVectorAddGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx = y.get_context();
        Tensor mat = ctx.get_input(0);
        Tensor vec = ctx.get_input(1);
        Tensor one = functional::constant(1, { y.shape()[0], 1 });
        Tensor dvec = functional::matmul(dy, one, true);
        mat.set_backprop_node(dy.get_node());
        mat.set_gradient(dy);
        vec.set_backprop_node(dvec.get_node());
        vec.set_gradient(dvec);
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

Tensor negative(Tensor x){
    OpAlgoContext ctx("Negative");
    Tensor y = create_variable(x.shape());
    ctx.add_input(x);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor add(Tensor x1, Tensor x2){
    Tensor y;
    auto x1_shape = x1.shape();
    auto x2_shape = x2.shape();
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
    
    if(max_dim == 4 && min_dim == 1){
        if(max_shape[1] == min_shape[0]){
            OpAlgoContext cxt("BatchedMatrixVectorAdd");
            y = functional::create_variable(max_shape);
            cxt.add_input(x1);
            cxt.add_input(x2);
            cxt.add_output(y);
            y.set_context(cxt);
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
            y = functional::create_variable(max_shape);
            cxt.add_input(x1);
            cxt.add_input(x2);
            cxt.add_output(y);
            y.set_context(cxt);
        }
        else{
            std::string err = "Can not add with ";
            err += std::to_string(max_dim) + "d";
            err += " and ";
            err += std::to_string(min_dim) + "d";
            throw err;
        }
    }
    else if(max_dim == min_dim && x1.size() == x2.size()){
        OpAlgoContext cxt("ElementwiseAdd");
        y = functional::create_variable(max_shape);
        cxt.add_input(x1);
        cxt.add_input(x2);
        cxt.add_output(y);
        y.set_context(cxt);
    }
    else {
        auto x1_shape = x1.shape();
        auto x2_shape = x2.shape();
        auto bc_shape = math::check_broadcasting(&x1_shape, &x2_shape);
        if (bc_shape.empty())
        {
            std::stringstream ss;
            ss << "Can not broadcast from ";
            for (auto& v : x1.shape()) ss << v << " ";
            ss << "to ";
            for (auto& v : x2.shape()) ss << v << " ";
            ss << std::endl;
            throw std::runtime_error(ss.str());
        }
        OpAlgoContext cxt("AddWithBroadcast");
        y = functional::create_variable(bc_shape);
        cxt.add_input(x1);
        cxt.add_input(x2);
        cxt.add_output(y);
        y.set_context(cxt);
    }

    return y;
}

Tensor sub(Tensor x1, Tensor x2){
    Tensor y = functional::create_variable(x1.shape());
    OpAlgoContext ctx("ElementwiseSub");
    ctx.add_input(x1);
    ctx.add_input(x2);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor mul(Tensor x1, Tensor x2){
    Tensor y = functional::create_variable(x1.shape());
    OpAlgoContext ctx("ElementwiseMul");
    ctx.add_input(x1);
    ctx.add_input(x2);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor div(Tensor x1, Tensor x2){
    Tensor y = functional::create_variable(x1.shape());
    OpAlgoContext ctx("ElementwiseDiv");
    ctx.add_input(x1);
    ctx.add_input(x2);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor add_n(std::vector<Tensor> xs){
    if(xs.size() >= 2){
        Tensor y = functional::create_variable(xs[0].shape());
        OpAlgoContext ctx("AddN");
        for(auto &x : xs){
            ctx.add_input(x);
        }
        ctx.add_output(y);
        y.set_context(ctx);
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
