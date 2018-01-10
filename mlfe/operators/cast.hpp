#ifndef __CAST_OP_HPP__
#define __CAST_OP_HPP__

#include "operator.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DeviceContext>
class CastOp final : public Operator<DeviceContext>{
public:
    explicit CastOp(
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                    std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                    ParamDef param
                    ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 1, "Input size must be 1(x).");
        runtime_assert(outputs.size() == 1, "Output size must be 1(y).");
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        
        if(this->GetParam().GetParamByName("Cast", cast) &&
           y->IsEmpty() &&
           !x->IsEmpty()
           ){
            if(!cast.compare("char")){
                y->template Resize<char>(x);
            }
            else if(!cast.compare("int")){
                y->template Resize<int>(x);
            }
            else if(!cast.compare("float")){
                y->template Resize<float>(x);
            }
            else if(!cast.compare("double")){
                y->template Resize<double>(x);
            }
            else{
                throw std::string("Wrong Type.");
            }
        }
        else{
            runtime_assert(x->Dims() == y->Dims(), "x's dim size must be same with y's dim.");
        }
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        if(!cast.compare("char")){
            TypeCaster<TypeLists<char, unsigned char, int, float, double>, char>::Run(x, y);
        }
        else if(!cast.compare("int")){
            TypeCaster<TypeLists<char, unsigned char, int, float, double>, int>::Run(x, y);
        }
        else if(!cast.compare("float")){
            TypeCaster<TypeLists<char, unsigned char, int, float, double>, float>::Run(x, y);
        }
        else if(!cast.compare("double")){
            TypeCaster<TypeLists<char, unsigned char, int, float, double>, double>::Run(x, y);
        }
        else{
            throw std::string("Wrong Type.");
        }
    }
    
protected:
    template <class ...Types>
    struct TypeLists{};
    
    template <class FirstType, class ...Types>
    struct TypeCaster;
    
    template <class FirstType, class ...Types, class ...To>
    struct TypeCaster<TypeLists<FirstType, Types...>, To...>{
        static void Run(
                        std::shared_ptr<TensorBlob<DeviceContext>> from,
                        std::shared_ptr<TensorBlob<DeviceContext>> to
                        ){
            if(from->template MatchType<FirstType>()){
                TypeCast<FirstType, To...>(from, to);
            }
            else{
                TypeCaster<TypeLists<Types...>, To...>::Run(from, to);
            }
        }
    };
    
    template <class ...To>
    struct TypeCaster<TypeLists<>, To...>{
        static void Run(
                        std::shared_ptr<TensorBlob<DeviceContext>> from,
                        std::shared_ptr<TensorBlob<DeviceContext>> to
                        ){
            throw std::string("No Type");
        }
    };
    
    template <class From, class To>
    static void TypeCast(
                  std::shared_ptr<TensorBlob<DeviceContext>> from,
                  std::shared_ptr<TensorBlob<DeviceContext>> to
                  ){
        const From *from_ptr = from->template GetPtrConst<From>();
        To *to_ptr = to->template GetPtrMutable<To>();
        for(int n = 0; n < from->Size(); ++n){
            to_ptr[n] = static_cast<To>(from_ptr[n]);
        }
    }
    
private:
    enum InputSchema{x};
    enum OutputSchema{y};
    std::string cast;
};

} /* namespace mlfe */
#endif /* __CAST_OP_HPP__ */
