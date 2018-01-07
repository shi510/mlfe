#include <iostream>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>
#include <mlfe/utils/db/simple_db.hpp>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>
#include <map>
#include <sstream>

using namespace std;
using namespace mlfe;
using namespace flatbuffers;

using TBC = TensorBlob<CPUContext>;

auto InverseEndian = [](int32_t data) -> int32_t{
    return (data >> 24 & 0x000000FF) | (data >> 8 & 0x0000FF00) | (data << 24 & 0xFF000000) | (data << 8 & 0x00FF0000);
};

flatbuffers::Offset<serializable::TensorBlob> MakeSerialiableTensor(
                                                                    flatbuffers::FlatBufferBuilder &fbb,
                                                                    string name,
                                                                    const uint8_t *data_ptr,
                                                                    vector<int> dim
                                                                    );

void SaveMNIST(shared_ptr<DataBase> db, string data_file_name, string label_file_name, TBC &data, TBC &label);
void VerifyDB(TBC &data, TBC &label, shared_ptr<DataBase> db);

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
    TBC training_data, training_label;
    TBC test_data, test_label;
    string path = args[1];
    
    try{
        sdb->option.delete_previous = true;
        sdb->option.binary = true;
        sdb->Open("mnist_train.simpledb");
        SaveMNIST(
                  sdb,
                  path + slash + string("train-images-idx3-ubyte"),
                  path + slash + string("train-labels-idx1-ubyte"),
                  training_data,
                  training_label
                  );
        sdb->Close();
        cout<<"training data done."<<endl;
        
        sdb->Open("mnist_test.simpledb");
        SaveMNIST(
                  sdb,
                  path + slash + string("t10k-images-idx3-ubyte"),
                  path + slash + string("t10k-labels-idx1-ubyte"),
                  test_data,
                  test_label
                  );
        sdb->Close();
        cout<<"testing data done."<<endl;
    }
    catch(string &e){
        cout<<e<<std::endl;
        return 0;
    }
    
    cout<<"Verifing DB...";
    try{
        sdb->option.delete_previous = false;
        sdb->Open("mnist_train.simpledb");
        VerifyDB(training_data, training_label, sdb);
        sdb->Close();
        
        sdb->Open("mnist_test.simpledb");
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

flatbuffers::Offset<serializable::TensorBlob>
MakeSerialiableTensor(flatbuffers::FlatBufferBuilder &fbb, string name, const uint8_t *data_ptr, vector<int> dim){
    int size = 1;
    for(auto &i : dim){ size *= i; }
    auto name_fb = fbb.CreateString(name);
    auto data_fb = fbb.CreateVector(data_ptr, size * sizeof(uint8_t));
    auto dim_fb = fbb.CreateVector(dim.data(), dim.size());
    auto tb_fb = serializable::CreateTensorBlob(fbb, name_fb, data_fb, dim_fb);
    return tb_fb;
}

void SaveMNIST(
               shared_ptr<DataBase> db,
               string data_file_name,
               string label_file_name,
               TBC &data,
               TBC &label
               ){
    int magic_num;
    int num_data;
    int num_label;
    int img_h;
    int img_w;
    int size;
    std::ifstream data_file, label_file;
    flatbuffers::FlatBufferBuilder fbb;
    
    data_file.open(data_file_name, std::ios::binary);
    label_file.open(label_file_name, std::ios::binary);
    if (!data_file.is_open()) {
        throw string("can not open file : ") + data_file_name;
    }
    if (!label_file.is_open()) {
        throw string("can not open file : ") + label_file_name;
    }
    data_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(InverseEndian(magic_num) != 2051){
        data_file.close();
        label_file.close();
        throw string("magic number dose not match.");
    }
    label_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(InverseEndian(magic_num) != 2049){
        data_file.close();
        label_file.close();
        throw string("magic number dose not match.");
    }
    data_file.read(reinterpret_cast<char *>(&num_data), 4);
    data_file.read(reinterpret_cast<char *>(&img_h), 4);
    data_file.read(reinterpret_cast<char *>(&img_w), 4);
    num_data = InverseEndian(num_data);
    img_h = InverseEndian(img_h);
    img_w = InverseEndian(img_w);
    label_file.read(reinterpret_cast<char *>(&num_label), 4);
    num_label = InverseEndian(num_label);
    if(num_data != num_label){
        data_file.close();
        label_file.close();
        throw string("number of data and number of label size are not match.");
    }
    
    size = img_h * img_w;
    data.Resize<uint8_t>({num_data, img_h, img_w});
    label.Resize<uint8_t>({num_label});
    for(int i = 0; i < num_data; ++i){
        std::vector<flatbuffers::Offset<serializable::TensorBlob>> tbs_std_vec;
        string tbs_str;
        data_file.read(data.GetPtrMutable<char>() + i * size, size);
        label_file.read(label.GetPtrMutable<char>() + i, 1);
        auto tb_data = MakeSerialiableTensor(fbb, "data" + to_string(i + 1), data.GetPtrConst<uint8_t>() + i * size, {img_h, img_w});
        auto tb_label = MakeSerialiableTensor(fbb, "label" + to_string(i + 1), label.GetPtrConst<uint8_t>() + i, {1});
        tbs_std_vec.push_back(tb_data);
        tbs_std_vec.push_back(tb_label);
        auto tbs_fb_vec = fbb.CreateVector(tbs_std_vec.data(), tbs_std_vec.size());
        auto tbs = serializable::CreateTensorBlobs(fbb, tbs_fb_vec);
        fbb.Finish(tbs);
        tbs_str.assign(reinterpret_cast<char *>(fbb.GetBufferPointer()), fbb.GetSize());
        db->Put(to_string(i + 1), tbs_str);
        fbb.Clear();
    }
    data_file.close();
    label_file.close();
}

void VerifyDB(TBC &data, TBC &label, shared_ptr<DataBase> db){
    flatbuffers::FlatBufferBuilder fbb;
    auto check_equal = [&data, &label](
                                       int cur_batch,
                                       const serializable::TensorBlob *data_tb_fb,
                                       const serializable::TensorBlob *label_tb_fb
                                       ){
        int size = data.Dim(1) * data.Dim(2);
        for(int i = 0; i < size; ++i){
            const char a = data.GetPtrConst<char>()[cur_batch * size + i];
            const char b = data_tb_fb->data()->Get(i);
            if(a != b){
                throw string("data not matches.");
            }
        }
        {
            const char a = label.GetPtrConst<char>()[cur_batch];
            const char b = label_tb_fb->data()->Get(0);
            if(a != b){
                throw string("label not matches.");
            }
        }
    };
    
    int batch_count = 0;
    db->MoveToFirst();
    do{
        string data_val;
        db->Get(data_val);
        auto tbs = serializable::GetTensorBlobs(data_val.data());
        check_equal(batch_count, tbs->tensors()->Get(0), tbs->tensors()->Get(1));
        ++batch_count;
    }while(db->MoveToNext());
    
    if(batch_count != data.Dim(0)){
        throw std::string("DB size does not match.");
    }
}
