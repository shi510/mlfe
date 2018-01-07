#ifndef __SIMPLE_MNIST_HPP__
#define __SIMPLE_MNIST_HPP__
#include "net_builder.hpp"

class SimpleMnist{
public:
    void SetDB(std::string train_db){
        this->train_db = train_db;
    }
    void Build(){
        OperatorInfo db_reader = builder.AddDBReader(
                                                     "input",
                                                     train_db,
                                                     "SimpleDB",
                                                     {100, 784},
                                                     100,
                                                     true
                                                     );
        std::string data = builder.AddCast("cast_data", db_reader.outputs[0], "float").outputs[0];
        std::string label = builder.AddCast("cast_label", db_reader.outputs[1], "float").outputs[0];
        std::string label_one_hot = builder.AddOneHot("onehot", label, 10).outputs[0];
        std::string prev_layer = data;
        prev_layer = builder.AddScale("scale", prev_layer, 1.f / 256.f).outputs[0];
        prev_layer = builder.AddFC("fc1", prev_layer, 10).outputs[0];
        prev_layer = builder.AddSoftmaxXent("softmax_xent", prev_layer, label_one_hot).outputs[0];
    }
    
    void Train(int iter, float lr){
        builder.Train(iter, lr);
    }
    
private:
    std::string train_db;
    NetBuilder builder;
};

#endif /* __SIMPLE_MNIST_HPP__ */
