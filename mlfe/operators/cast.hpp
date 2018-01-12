#ifndef __CAST_OP_HPP__
#define __CAST_OP_HPP__
#include "operator.hpp"

namespace mlfe{

template <class DeviceContext>
class CastOp final : public Operator<DeviceContext>{
public:
    CastOp(OperatorIO &opio, ItemHolder *ih);
    
    void Compute() override;
    
protected:
    template <class ...Types>
    struct TypeLists{};
    
    template <class FirstType, class ...Types>
    struct TypeCaster;
    
    template <class FirstType, class ...Types, class ...To>
    struct TypeCaster<TypeLists<FirstType, Types...>, To...>{
        static void Run(
                        TensorBlob<DeviceContext> *from,
                        TensorBlob<DeviceContext> *to
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
                        TensorBlob<DeviceContext> *from,
                        TensorBlob<DeviceContext> *to
                        ){
            throw std::string("No Type");
        }
    };
    
    template <class From, class To>
    static void TypeCast(
                  TensorBlob<DeviceContext> *from,
                  TensorBlob<DeviceContext> *to
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
