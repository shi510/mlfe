#include "dense.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

REGIST_OP(Dense)
    .Input("X", "float32")
    .Input("W", "float32")
    .Input("B", "float32")
    .Output("Y", "float32")
    .Attr("output", "int32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto w = odc->Input(1);
        auto b = odc->Input(2);
        auto y = odc->Output(0);
        auto n = odc->GetAttr<type::int32::T>("output");
        w.Reshape({ n, x.Shape()[1] }, type::float32());
        b.Reshape({ n }, type::float32());
        y.Reshape({ x.Shape()[0], n }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(Dense)
    .Input("X", "float32")
    .Input("W", "float32")
    .Input("B", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dW", "float32")
    .Output("dB", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto w = odc->Input(1);
        auto b = odc->Input(2);
        auto dw = odc->Output(0);
        auto db = odc->Output(1);
        auto dx = odc->Output(2);
        dw.Reshape(w.Shape(), type::float32());
        db.Reshape(b.Shape(), type::float32());
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class DenseGradient : public GradientHelper{
public:
    DenseGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = y.get_children()[0];
        Tensor w = y.get_children()[1];
        Tensor b = y.get_children()[2];
        Tensor dw, db, dx;

        dep = OpDependency::Builder("DenseGradient")
            .Input(x).Input(w).Input(b).Input(y).Input(dy)
            .Output(dw).Output(db).Output(dx)
            .Finish();

        gpair[x] = dx;
        gpair[w] = dw;
        gpair[b] = db;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Dense, DenseGradient)

Tensor Dense(Tensor x, type::int32::T num_out, Tensor init_w, Tensor init_b){
    Tensor w, b, y;

    w.Initialize(init_w);
    b.Initialize(init_b);
    w.set_trainable(true);
    b.set_trainable(true);

    auto dep = OpDependency::Builder("Dense")
        .Input(x).Input(w).Input(b)
        .Output(y)
        .Attr({ "output", num_out })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    y.add_child(x);
    y.add_child(w);
    y.add_child(b);

    return y;
}

} // end namespace functional
} // end namespace mlfe
