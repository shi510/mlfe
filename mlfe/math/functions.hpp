#ifndef __MATH__FUNCTIONS_HPP__
#define __MATH__FUNCTIONS_HPP__
#include "../device_context/cpu_context.hpp"

namespace mlfe{ namespace math{

template <class DataType, class DeviceContext>
void ReluFunction(const int size, const DataType *x, DataType *y);

template <class DataType, class DeviceContext>
void ReluGradientFunction(const int size, const DataType *y, const DataType *dy, DataType *dx);

} /* namespace math */
} /* namespace mlfe */
#endif
