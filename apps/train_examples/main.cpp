#include <iostream>
#include <memory>
#include <iomanip>

#include "net_builder.hpp"

using namespace mlfe;

int main(int argc, char *argv[]){
    NetBuilder net;
    if(argc < 2){
        std::cout<<"mnist data folder path must be fed into argument."<<std::endl;
        return -1;
    }
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    const std::string slash="\\";
#else
    const std::string slash="/";
#endif
    try{
        OperatorInfo db_reader = net.AddDBReader(
                                                 "input",
                                                 argv[1] + slash + "mnist_train.simpledb",
                                                 "SimpleDB",
                                                 {100, 784},
                                                 100,
                                                 true,
                                                 true
                                                 );
        std::string data = net.AddCast("cast_data", db_reader.outputs[0], "float").outputs[0];
        std::string label = net.AddCast("cast_label", db_reader.outputs[1], "float").outputs[0];
        std::string label_one_hot = net.AddOneHot("onehot", label, 10).outputs[0];
        std::string prev_layer = data;
        prev_layer = net.AddScale("scale", prev_layer, 1.f / 256.f).outputs[0];
        prev_layer = net.AddFC("fc1", prev_layer, 10).outputs[0];
        prev_layer = net.AddSoftmaxXent("softmax_xent", prev_layer, label_one_hot).outputs[0];
        net.Train(1000, 0.5);
    }
    catch(std::string &e){
        std::cout<<e<<std::endl;
        return -1;
    }
    return 0;
}

