#include <iostream>
#include "vgg16.h"

using namespace mlfe;

int main(int argc, char *argv[])
{
    try{
        models::vgg16 net;
        auto input = Tensor::from_vector<float>(std::vector<float>(224*224*3), {16, 224, 224, 3});
        for(int n = 0; n < 10; ++n){
            auto output = net.forward(input, true);
            std::cout<<output.data<float>()[0]<<std::endl;
        }
    }
    catch(std::exception &e){
        std::cout<<e.what()<<std::endl;
    }

    return 0;
}

