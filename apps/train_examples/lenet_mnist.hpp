#ifndef __LENET_MNIST_HPP__
#define __LENET_MNIST_HPP__
#include "net_builder.hpp"

class LenetMnist{
public:
    void SetDB(std::string train_db){
        this->train_db = train_db;
    }
    void Build(){
        OperatorInfo db_reader = builder.AddDBReader(
                                                     "input",
                                                     train_db,
                                                     "SimpleDB",
                                                     {60, 1, 28, 28},
                                                     60,
                                                     true
                                                     );
        std::string data = builder.AddCast("cast_data", db_reader.outputs[0], "float").outputs[0];
        std::string label = builder.AddCast("cast_label", db_reader.outputs[1], "float").outputs[0];
        std::string label_one_hot = builder.AddOneHot("onehot", label, 10).outputs[0];
        std::string prev_layer = data;
        prev_layer = builder.AddScale("scale", prev_layer, 1.f / 256.f).outputs[0];
        prev_layer = builder.AddConv("conv1", prev_layer, 20, {5, 5}, {1, 1}, 0).outputs[0];
        prev_layer = builder.AddMaxPool("maxpool1", prev_layer, {2, 2}, {2, 2}).outputs[0];
        prev_layer = builder.AddConv("conv2", prev_layer, 50, {5, 5}, {1, 1}, 0).outputs[0];
        prev_layer = builder.AddMaxPool("maxpool2", prev_layer, {2, 2}, {2, 2}).outputs[0];
        prev_layer = builder.AddFlatten("flatten", prev_layer, 1).outputs[0];
        prev_layer = builder.AddFC("fc1", prev_layer, 500).outputs[0];
        prev_layer = builder.AddRelu("relu", prev_layer, true).outputs[0];
        prev_layer = builder.AddFC("fc2", prev_layer, 10).outputs[0];
        prev_layer = builder.AddSoftmaxXent("softmax_xent", prev_layer, label_one_hot).outputs[0];
    }
    
    void Train(int iter, float lr){
        builder.Train(iter, lr);
    }
    
private:
    std::string train_db;
    NetBuilder builder;
};

#endif /* __LENET_MNIST_HPP__ */
