#ifndef __SIMPLEDB_MNIST_H__
#define __SIMPLEDB_MNIST_H__
#include <vector>
#include <string>
#include <memory>
#include <mlfe/utils/db/simple_db.h>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>

void CreateSimpledbForMnist(std::string train_simpledb_name,
                            std::string test_simpledb_name,
                            std::string train_data_path, 
                            std::string train_label_path, 
                            std::string test_data_path,
                            std::string test_label_path
                           );

#endif // end #ifndef __SIMPLEDB_MNIST_H__