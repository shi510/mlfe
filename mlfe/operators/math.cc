#include "math.h"
#include "../core/graph.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"

namespace mlfe{
namespace functional{

REGIST_OP(SquaredDifference)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", "float32")
    .Attr("output", "int32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto x2 = odc->Input(1);
        auto y = odc->Output(0);
        
        if(x1.Dims() != x2.Dims()){
            throw std::string("squared_difference op : ") +
                std::string("x1.Shape() != x2.Shape()");
        }
        for(int n = 0; n < x1.Dims(); ++n){
            if(x1.Shape()[n] != x2.Shape()[n]){
                throw std::string("squared_difference op : ") +
                    std::string("x1.Shape() != x2.Shape()");
            }
        }
        y.Reshape(x1.Shape(), type::float32());
    })
    .Finish();

REGIST_OP_GRAD(SquaredDifference)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Input("dY", "float32")
    .Output("dX1", "float32")
    .Output("dX2", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto x2 = odc->Input(1);
        auto dy = odc->Input(2);
        auto dx1 = odc->Output(0);
        auto dx2 = odc->Output(1);
        dx1.Reshape(x1.Shape(), type::float32());
        dx2.Reshape(x2.Shape(), type::float32());
    })
    .Finish();

class SquaredDifferenceGradient : public GradientHelper{
public:
    SquaredDifferenceGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y,
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x1 = y.get_children()[0];
        Tensor x2 = y.get_children()[1];
        Tensor dx1, dx2;

        dep = OpDependency::Builder("SquaredDifferenceGradient")
            .Input(x1).Input(x2).Input(dy)
            .Output(dx1).Output(dx2)
            .Finish();

        gpair[x1] = dx1;
        gpair[x2] = dx2;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(SquaredDifference, SquaredDifferenceGradient)

Tensor squared_difference(Tensor x1, Tensor x2){
    Tensor y;

    auto dep = OpDependency::Builder("SquaredDifference")
        .Input(x1).Input(x2)
        .Output(y)
        .Finish();

    y = Tensor::DependencyAdder(dep);

    y.add_child(x1);
    y.add_child(x2);

    return y;
}

} // end namespace functional
} // end namespace mlfe
