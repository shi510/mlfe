#include <iostream>
#include <memory>
#include <iomanip>

#include "simple_mnist.hpp"
#include "lenet_mnist.hpp"

using namespace mlfe;

int main(int argc, char *argv[]){
    if(argc < 3){
        std::cout<<"mnist data folder path and net type must be fed into argument."<<std::endl;
        std::cout<<"ex)"<<std::endl;
        std::cout<<"        simple /usr/home/mnist_train.simpledb"<<std::endl;
        std::cout<<"        lenet /usr/home/mnist_train.simpledb"<<std::endl;
        return -1;
    }
    try{
        if(!std::string(argv[1]).compare("simple")){
            SimpleMnist simple;
            simple.SetDB(argv[2]);
            simple.Build();
            simple.Train(1000, 0.5f);
        }
        else if(!std::string(argv[1]).compare("lenet")){
            LenetMnist simple;
            simple.SetDB(argv[2]);
            simple.Build();
            simple.Train(4000, 0.02f);
        }
    }
    catch(std::string &e){
        std::cout<<e<<std::endl;
        return -1;
    }
    return 0;
}
