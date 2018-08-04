#ifndef __MATH__FUNCTIONS_HPP__
#define __MATH__FUNCTIONS_HPP__

namespace mlfe{ namespace math{

template <class DataType, class DeviceContext>
void ReluFunction(const int size, const DataType *x, DataType *y);

template <class DataType, class DeviceContext>
void ReluGradientFunction(const int size, const DataType *x, const DataType *dy, DataType *dx);

template <class DataType, class DeviceContext>
void SigmoidFunction(const int size, const DataType *x, DataType *y);

template <class DataType, class DeviceContext>
void SigmoidGradientFunction(const int size, const DataType *y, const DataType *dy, DataType *dx);

unsigned int GetRandomSeed();

} /* namespace math */
} /* namespace mlfe */
#endif
