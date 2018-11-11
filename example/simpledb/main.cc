#include "cifar.h"
#include "mnist.h"
#include "wider_face.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cctype>
#include <algorithm>

std::string UsageString();

int main(int argc, char *args[]){
    if(argc < 3){
        std::cout << UsageString() << std::endl;
        return 0;
    }
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    const std::string slash="\\";
#else
    const std::string slash="/";
#endif
    std::string data_name(args[1]);

    std::transform(data_name.begin(), data_name.end(), data_name.begin(),
                   [](unsigned char c){return std::tolower(c); });

    try{
        if(!data_name.compare("mnist")){
            CreateSimpledbForMnist(
                "mnist_train.simpledb",
                "mnist_test.simpledb",
                args[2] + slash + std::string("train-images-idx3-ubyte"),
                args[2] + slash + std::string("train-labels-idx1-ubyte"),
                args[2] + slash + std::string("t10k-images-idx3-ubyte"),
                args[2] + slash + std::string("t10k-labels-idx1-ubyte")
            );
        }
        else if(!data_name.compare("cifar10")){
            CreateSimpledbForCifar(
                "cifar10_train.simpledb",
                "cifar10_test.simpledb",
                {
                    args[2] + slash + std::string("data_batch_1.bin"),
                    args[2] + slash + std::string("data_batch_2.bin"),
                    args[2] + slash + std::string("data_batch_3.bin"),
                    args[2] + slash + std::string("data_batch_4.bin"),
                    args[2] + slash + std::string("data_batch_5.bin")
                },
                args[2] + slash + std::string("test_batch.bin")
            );
        }
        else if(!data_name.compare("wider_face")){
            CreateSimpledbForWiderFace(
                "wider_face_pos_train.simpledb",
                "wider_face_part_train.simpledb",
                "wider_face_neg_train.simpledb",
                args[2],
                args[3] + slash + "wider_face_train_bbx_gt.txt"
            );
        }
        else{
            std::cout << "You feed wrong data name, you typed \"" << data_name <<"\""<< std::endl;
            std::cout << UsageString() << std::endl;
        }
    }
    catch(std::string &e){
        std::cout << e << std::endl;
        return 1;
    }

    return 0;
}

std::string UsageString(){
    std::stringstream ss;
    ss << "Usage : ";
    ss << "./simpledb [mnist | cifar10] [path] " << std::endl;
    ss << "    " << "the [path] must include files for mnist as below :" << std::endl;
    ss << "        " << "train-images-idx3-ubyte" << std::endl;
    ss << "        " << "train-labels-idx1-ubyte" << std::endl;
    ss << "        " << "t10k-images-idx3-ubyte" << std::endl;
    ss << "        " << "t10k-labels-idx1-ubyte" << std::endl;
    ss << "    " << "the [path] must include files for cifar10 as below :" << std::endl;
    ss << "        " << "data_batch_1.bin" << std::endl;
    ss << "        " << "data_batch_2.bin" << std::endl;
    ss << "        " << "data_batch_3.bin" << std::endl;
    ss << "        " << "data_batch_4.bin" << std::endl;
    ss << "        " << "data_batch_5.bin" << std::endl;
    ss << "        " << "test_batch.bin" << std::endl;
    return ss.str();
}
