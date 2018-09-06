#ifndef __SIMPLEDB_CIFAR_H__
#define __SIMPLEDB_CIFAR_H__
#include <vector>
#include <string>

void CreateSimpledbForCifar(std::string train_simpledb_name,
                            std::string test_simpledb_name,
                            std::vector<std::string> train_batch_paths, 
                            std::string test_batch_path
                           );

#endif // end #ifndef __SIMPLEDB_CIFAR_H__