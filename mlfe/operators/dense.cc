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
        auto x = odc->Input("X");
        auto w = odc->Input("W");
        auto b = odc->Input("B");
        auto y = odc->Output("Y");
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
        auto x = odc->Input("X");
        auto w = odc->Input("W");
        auto b = odc->Input("B");
        auto dw = odc->Output("dW");
        auto db = odc->Output("dB");
        auto dx = odc->Output("dX");
        dw.Reshape(w.Shape(), type::float32());
        db.Reshape(b.Shape(), type::float32());
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class DenseGradient : public GradientHelper{
public:
    DenseGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor w = odc->Input("W");
        Tensor b = odc->Input("B");
        Tensor y = odc->Output("Y");
        Tensor dw, db, dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("DenseGradient")
            .Input(std::make_tuple("X", x))
            .Input(std::make_tuple("W", w))
            .Input(std::make_tuple("B", b))
            .Input(std::make_tuple("Y", y))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("W"), dw))
            .Output(std::make_tuple(Gradient("B"), db))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Finish();

        dx = Tensor::DependencyAdder(dep);
        pairs.push_back({ w, dw });
        pairs.push_back({ b, db });

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(Dense, DenseGradient)

Tensor Dense(Tensor x, type::int32::T num_out, Tensor init_w, Tensor init_b){
    Tensor w, b, y;

    w.Initialize(init_w);
    b.Initialize(init_b);
    auto dep = OpDependency::Builder("Dense")
        .Input(std::make_tuple("X", x))
        .Input(std::make_tuple("W", w))
        .Input(std::make_tuple("B", b))
        .Output(std::make_tuple("Y", y))
        .Attr({ "output", num_out })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}

} // end namespace functional
} // end namespace mlfe
