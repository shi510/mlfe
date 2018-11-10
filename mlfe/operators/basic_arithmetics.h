#ifndef __BASIC_ARITHMETICS_OP_HPP__
#define __BASIC_ARITHMETICS_OP_HPP__
#include "../core/tensor.h"

namespace mlfe{ namespace functional{

// Elementwise Addition
template <class T, 
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor Add(Tensor a, T b);

// Elementwise Addition
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor Sub(Tensor a, T b);

// Elementwise Addition
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor Mul(Tensor a, T b);

// Elementwise Addition
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor Div(Tensor a, T b);

Tensor negative(Tensor x);

Tensor add(Tensor x1, Tensor x2);

Tensor sub(Tensor x1, Tensor x2);

Tensor mul(Tensor x1, Tensor x2);

Tensor div(Tensor x1, Tensor x2);

Tensor add_n(std::vector<Tensor> xs);

} // end namespace functional

// Elementwise Addition
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor operator+(Tensor a, T b);

// Elementwise Substraction
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor operator-(Tensor a, T b);

// Elementwise Multiplication
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor operator*(Tensor a, T b);

// Elementwise Division
template <class T,
    typename = typename std::enable_if<
    std::is_same<Tensor, T>::value ||
    std::is_fundamental<T>::value>::type
>
Tensor operator/(Tensor a, T b);

} // end namespace mlfe
#endif // end ifndef __BASIC_ARITHMETICS_OP_HPP__
