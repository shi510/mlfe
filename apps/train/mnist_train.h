#ifndef __EXAMPLE_MNIST_TRAIN_H__
#define __EXAMPLE_MNIST_TRAIN_H__
#include <mlfe/core/tensor.h>
#include <string>

namespace train_example{

void train_simple_mnist(const std::string train_path,
                        const std::string test_path,
                        const int batch,
                        const int iter,
                        const double lr,
                        const double mm
                       );

void train_lenet(const std::string train_path,
                 const std::string test_path,
                 const int batch,
                 const int iter,
                 const double lr,
                 const double mm
                );

void train_ae(const std::string train_path,
              const std::string test_path,
              const int batch,
              const int iter,
              const double lr,
              const double mm
             );

} // end namespace train_example
#endif // __EXAMPLE_MNIST_TRAIN_H__