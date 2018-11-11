#include "mnist_train.h"
#include <iostream>
#include <string>
#include <sstream>

int main(int argc, char *argv[]){
    auto parse_file_name = [](std::string str) -> std::string {
        std::string file_name;
        for(int n = str.size() - 1; n >= 0; --n){
            if(str[n] == '\\' || str[n] == '/'){
                break;
            }
            file_name.append(1, str[n]);
        }
        std::reverse(file_name.begin(), file_name.end());
        return file_name;
    };
    const std::vector<std::string> ex_list = {
        "mnist_simple",
        "mnist_lenet",
        "mnist_autoencoder"
    };
    const auto file_name = parse_file_name(argv[0]);

    if(argc == 1){
        std::stringstream ss;
        ss << "Usage : " << std::endl;
        ss << "    " << file_name << " [ ";
        for(auto ex : ex_list){
            ss << ex << " | ";
        }
        std::string usage_msg = ss.str();
        usage_msg.erase(usage_msg.size() - 3, 3);
        usage_msg.append("]");
        std::cout << usage_msg << std::endl;
        return 0;
    }

    try{
        const std::string ex_name = argv[1];
        if(ex_name == "mnist_simple"){
            if(argc < 4){
                std::cout << "Usage : " << std::endl;
                std::cout <<"    "<< file_name << " " << "mnist_simple";
                std::cout << " [mnist_train_db_path]";
                std::cout << " [mnist_test_db_path]" << std::endl;
                return 0;
            }
            std::string mnist_train_path = argv[2];
            std::string mnist_test_path = argv[3];
            train_example::train_simple_mnist(mnist_train_path,
                                              mnist_test_path,
                                              64, // batch
                                              1000, // iteration
                                              1e-1, // learning rate
                                              0 // momentum
                                             );
        }
        else if(ex_name == "mnist_lenet"){
            if(argc < 4){
                std::cout << "Usage : " << std::endl;
                std::cout << "    " << file_name << " " << "mnist_lenet";
                std::cout << " [mnist_train_db_path]";
                std::cout << " [mnist_test_db_path]" << std::endl;
                return 0;
            }
            std::string mnist_train_path = argv[2];
            std::string mnist_test_path = argv[3];
            train_example::train_lenet(mnist_train_path,
                                       mnist_test_path,
                                       64, // batch
                                       4000, // iteration
                                       1e-2, // learning rate
                                       0.9 // momentum
                                      );
        }
        else if(ex_name == "mnist_autoencoder"){
            if(argc < 4){
                std::cout << "Usage : " << std::endl;
                std::cout << "    " << file_name << " " << "mnist_autoencoder";
                std::cout << " [mnist_train_db_path]";
                std::cout << " [mnist_test_db_path]" << std::endl;
                return 0;
            }
            std::string mnist_train_path = argv[2];
            std::string mnist_test_path = argv[3];
            train_example::train_ae(mnist_train_path,
                                    mnist_test_path,
                                    64, // batch
                                    65000, // iteration
                                    1e-1, // learning rate
                                    0.9 // momentum
                                   );
        }
        else{
            std::cout << "wrong input." << std::endl;
            return 0;
        }
        
    }
    catch(std::string &e){
        std::cout << "[ERROR] : " << e << std::endl;
    }
    return 0;
}
