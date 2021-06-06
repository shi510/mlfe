#include "mnist.h"
#include <fstream>
#include <sstream>

namespace dataset
{

int32_t inverse_endian(int32_t data){
    return
        (data >> 24 & 0x000000FF) |
        (data >>  8 & 0x0000FF00) |
        (data << 24 & 0xFF000000) |
        (data <<  8 & 0x00FF0000);
};

void read_header(std::ifstream &data_file,
    std::ifstream &label_file,
    int &num_data, int &h, int &w)
{
    int magic_num;
    int num_label;

    data_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(inverse_endian(magic_num) != 2051){
        data_file.close();
        label_file.close();
        throw std::string("magic number dose not match.");
    }
    label_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(inverse_endian(magic_num) != 2049){
        data_file.close();
        label_file.close();
        throw std::string("magic number dose not match.");
    }
    data_file.read(reinterpret_cast<char *>(&num_data), 4);
    data_file.read(reinterpret_cast<char *>(&h), 4);
    data_file.read(reinterpret_cast<char *>(&w), 4);
    num_data = inverse_endian(num_data);
    h = inverse_endian(h);
    w = inverse_endian(w);
    label_file.read(reinterpret_cast<char *>(&num_label), 4);
    num_label = inverse_endian(num_label);
    if(num_data != num_label){
        data_file.close();
        label_file.close();
        throw std::string("number of data and number of label size are not match.");
    }
}

void read_all(std::string data_file_name,
    std::string label_file_name,
    std::vector<uint8_t> &data,
    std::vector<uint8_t> &label)
{
    int num_data;
    int img_h;
    int img_w;
    int size;
    std::ifstream data_file, label_file;

    data_file.open(data_file_name, std::ios::binary);
    label_file.open(label_file_name, std::ios::binary);
    if(!data_file.is_open()) {
        throw std::string("can not open file : ") + data_file_name;
    }
    if(!label_file.is_open()) {
        throw std::string("can not open file : ") + label_file_name;
    }
    read_header(data_file, label_file, num_data, img_h, img_w);

    size = img_h * img_w;
    data.resize(num_data * size);
    label.resize(num_data);
    for(int n = 0; n < num_data; ++n){
        std::string tbs_str;
        data_file.read((char *)data.data() + n * size, size);
        label_file.read((char *)label.data() + n, 1);
    }
    data_file.close();
    label_file.close();
}

void read_mnist_dataset(std::string folder_path,
    std::vector<uint8_t>& train_x,
    std::vector<uint8_t>& train_y,
    std::vector<uint8_t>& valid_x,
    std::vector<uint8_t>& valid_y)
{
    read_all(
        folder_path + "/train-images-idx3-ubyte",
        folder_path + "/train-labels-idx1-ubyte",
        train_x,
        train_y
    );

    read_all(
        folder_path + "/t10k-images-idx3-ubyte",
        folder_path + "/t10k-labels-idx1-ubyte",
        valid_x,
        valid_y
    );
}

} // end namespace dataset