#include <iostream>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/utils/db/simple_db.hpp>
#include <map>
#include <sstream>

using namespace std;
using namespace mlfe;

auto InverseEndian = [](int32_t data) -> int32_t{
    return (data >> 24 & 0x000000FF) | (data >> 8 & 0x0000FF00) | (data << 24 & 0xFF000000) | (data << 8 & 0x00FF0000);
};

void DataGetter(string file_path, TensorBlob<CPUContext> &data, shared_ptr<DataBase> db);
void LabelGetter(string file_path, TensorBlob<CPUContext> &label, shared_ptr<DataBase> db);
void VerifyDB(TensorBlob<CPUContext> &data, TensorBlob<CPUContext> &label, shared_ptr<DataBase> db);

int main(int argc, char *args[]){
    if(argc < 2){
        cout<<"MNIST Data folder path must be fed."<<endl;
        return 0;
    }
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    const std::string slash="\\";
#else
    const std::string slash="/";
#endif
    cout<<"Your MNIST data folder : "<<args[1]<<endl;
    shared_ptr<DataBase> sdb = make_shared<simpledb::SimpleDB>();
    TensorBlob<CPUContext> training_data, training_label;
    TensorBlob<CPUContext> test_data, test_label;
    string path = args[1];
    
    try{
        string data_str, label_str;
        
        sdb->Open("training_data.simpledb");
        DataGetter(path + slash + string("train-images-idx3-ubyte"), training_data, sdb);
        LabelGetter(path + slash + string("train-labels-idx1-ubyte"), training_label, sdb);
        sdb->Close();
        cout<<"training data done."<<endl;
        
        sdb->Open("test_data.simpledb");
        DataGetter(path + slash + string("t10k-images-idx3-ubyte"), test_data, sdb);
        LabelGetter(path + slash + string("t10k-labels-idx1-ubyte"), test_label, sdb);
        sdb->Close();
        cout<<"testing data done."<<endl;
    }
    catch(string &e){
        cout<<e<<std::endl;
        return 0;
    }
    
    cout<<"Verifing DB...";
    try{
        string data_str, label_str;
        
        sdb->Open("training_data.simpledb");
        VerifyDB(training_data, training_label, sdb);
        sdb->Close();
        
        sdb->Open("test_data.simpledb");
        VerifyDB(test_data, test_label, sdb);
        sdb->Close();
    }
    catch(string &e){
        cout<<e<<std::endl;
        return 0;
    }
    cout<<"Done."<<endl;
    return 1;
}

void DataGetter(string file_path, TensorBlob<CPUContext> &data, shared_ptr<DataBase> db){
    int magic_num;
    int num_data;
    int img_h;
    int img_w;
    int size;
    std::ifstream mnist_file;
    
    mnist_file.open(file_path, std::ios::binary);
    if (!mnist_file.is_open()) {
        throw string("can not open file : ") + file_path;
    }
    mnist_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(InverseEndian(magic_num) != 2051){
        throw "magic number dose not match.";
    }
    mnist_file.read(reinterpret_cast<char *>(&num_data), 4);
    mnist_file.read(reinterpret_cast<char *>(&img_h), 4);
    mnist_file.read(reinterpret_cast<char *>(&img_w), 4);
    num_data = InverseEndian(num_data);
    img_h = InverseEndian(img_h);
    img_w = InverseEndian(img_w);
    data.Reshape<char>({num_data, img_h, img_w});
    size = img_h * img_w;
    for(int i = 0; i < data.Dim(0); ++i){
        string to_str;
        mnist_file.read(data.GetPtrMutable<char>() + i * size, size);
        to_str.assign(data.GetPtrConst<char>() + i * size, size);
        db->Put("data_" + to_string(i + 1), to_str);
    }
    mnist_file.close();
}

void LabelGetter(string file_path, TensorBlob<CPUContext> &label, shared_ptr<DataBase> db){
    int magic_num;
    int num_data;
    std::ifstream mnist_file;
    
    mnist_file.open(file_path, std::ios::binary);
    if (!mnist_file.is_open()) {
        throw string("can not open file : ") + file_path;
    }
    mnist_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(InverseEndian(magic_num) != 2049){
        throw "magic number dose not match.";
    }
    mnist_file.read(reinterpret_cast<char *>(&num_data), 4);
    num_data = InverseEndian(num_data);
    label.Reshape<char>({num_data});
    for(int i = 0; i < label.Dim(0); ++i){
        string to_str;
        mnist_file.read(label.GetPtrMutable<char>() + i, 1);
        to_str.assign(label.GetPtrConst<char>() + i, 1);
        db->Put("label_" + to_string(i + 1), to_str);
    }
    mnist_file.close();
}

void VerifyDB(TensorBlob<CPUContext> &data, TensorBlob<CPUContext> &label, shared_ptr<DataBase> db){
    auto check_equal = [&data, &label](int batch, string &data_val, string &label_val){
        for(int i = 0; i < data.Dim(1) * data.Dim(2); ++i){
            const char a = data.GetPtrConst<char>()[batch * data.Dim(1) * data.Dim(2) + i];
            const char b = reinterpret_cast<const char *>(data_val.data())[i];
            if(a != b){
                throw string("value not matches.");
            }
        }
        {
            const char a = label.GetPtrConst<char>()[batch];
            const char b = reinterpret_cast<const char *>(label_val.data())[0];
            if(a != b){
                throw string("value not matches.");
            }
        }
    };
    
    for(int i = 0; i < data.Dim(0); ++i){
        string data_val, label_val;
        db->Get("data_" + to_string(i + 1), data_val);
        db->Get("label_" + to_string(i + 1), label_val);
        check_equal(i, data_val, label_val);
    }
}
