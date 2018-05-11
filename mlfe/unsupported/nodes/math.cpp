#include "math.hpp"
#include "../../math/blas.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

//TODO : 
// make more simple on the two cases that are: 
// 1. one input, one value and one output. (C_Tensor = A_Tensor [BinaryOpExpr] Constant_Value)
// 2. two input and one output. (C_Tensor = A_Tensor [BinaryOpExpr] B_Tensor)
#define DEFINE_BIANRY_OP_NODE(OpName)\
OpName::OpName() : NodeSchema<OpName>(#OpName) { }\
void OpName::InternalInit(Workspace *ws, OperatorContext *oc) {\
    runtime_assert(Inputs() == 1 || Inputs() == 2,\
        std::string("Inputs must be 1(a) or 2(a, b).") +\
        std::string(" - Your input size : ") +\
        std::to_string(Inputs())\
    );\
    runtime_assert(Outputs() == 1,\
        std::string("Outputs must be 1(c).") +\
        std::string(" - Your output size : ") +\
        std::to_string(Outputs())\
    );\
    Node *base = reinterpret_cast<Node *>(this);\
    Tensor *a = ws->Get<Tensor>(base->Input(0));\
    Tensor *c = ws->GetIfNotExistCreate<Tensor>(base->Output(0));\
    Tensor *b = nullptr;\
    if(Inputs() == 2){\
        b = ws->Get<Tensor>(base->Input(1));\
        if (a->Dims() != b->Dims()) {\
                throw std::string("Add : inputs are not same Dims. : ") + base->Input(0) + base->Input(1); \
        }\
    }\
    if(c->Dims() == 0) {\
        std::vector<int> dim;\
        for (int i = 0; i < a->Dims(); ++i) {\
            dim.push_back(a->Dim(i));\
        }\
        c->Reshape(dim);\
    }\
    oc->inputs.push_back(a);\
    if(b != nullptr){\
        oc->inputs.push_back(b);\
    }\
    oc->outputs.push_back(c);\
}\
void OpName::InternalGradientInit(Workspace *ws, OperatorContext *oc) { }

DEFINE_BIANRY_OP_NODE(Add)
DEFINE_BIANRY_OP_NODE(Sub)
DEFINE_BIANRY_OP_NODE(Mul)
DEFINE_BIANRY_OP_NODE(Div)

} // end namespace node
} // end namespace mlfe
