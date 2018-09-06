#include "mnist.h"
#include <map>
#include <sstream>

int32_t InverseEndian(int32_t data){
    return
        (data >> 24 & 0x000000FF) |
        (data >>  8 & 0x0000FF00) |
        (data << 24 & 0xFF000000) |
        (data <<  8 & 0x00FF0000);
};

void ReadMnistHeader(std::ifstream &data_file,
                     std::ifstream &label_file,
                     int &num_data, int &h, int &w
                    )
{
    int magic_num;
    int num_label;

    data_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(InverseEndian(magic_num) != 2051){
        data_file.close();
        label_file.close();
        throw std::string("magic number dose not match.");
    }
    label_file.read(reinterpret_cast<char *>(&magic_num), 4);
    if(InverseEndian(magic_num) != 2049){
        data_file.close();
        label_file.close();
        throw std::string("magic number dose not match.");
    }
    data_file.read(reinterpret_cast<char *>(&num_data), 4);
    data_file.read(reinterpret_cast<char *>(&h), 4);
    data_file.read(reinterpret_cast<char *>(&w), 4);
    num_data = InverseEndian(num_data);
    h = InverseEndian(h);
    w = InverseEndian(w);
    label_file.read(reinterpret_cast<char *>(&num_label), 4);
    num_label = InverseEndian(num_label);
    if(num_data != num_label){
        data_file.close();
        label_file.close();
        throw std::string("number of data and number of label size are not match.");
    }
}

flatbuffers::Offset<mlfe::serializable::TensorBlob>
MakeSerialiableTensor(flatbuffers::FlatBufferBuilder &fbb,
                      std::string name,
                      const uint8_t *data_ptr,
                      std::vector<int> dim
                     )
{
    int size = 1;
    for(auto &i : dim){ size *= i; }
    auto name_fb = fbb.CreateString(name);
    auto data_fb = fbb.CreateVector(data_ptr, size * sizeof(uint8_t));
    auto dim_fb = fbb.CreateVector(dim.data(), dim.size());
    auto tb_fb = mlfe::serializable::CreateTensorBlob(fbb, name_fb, data_fb, dim_fb);
    return tb_fb;
}

void SaveMNIST(std::shared_ptr<mlfe::DataBase> db,
               std::string data_file_name,
               std::string label_file_name,
               std::vector<unsigned char> &data,
               std::vector<unsigned char> &label
              )
{
    int num_data;
    int img_h;
    int img_w;
    int size;
    std::ifstream data_file, label_file;
    flatbuffers::FlatBufferBuilder fbb;

    data_file.open(data_file_name, std::ios::binary);
    label_file.open(label_file_name, std::ios::binary);
    if(!data_file.is_open()) {
        throw std::string("can not open file : ") + data_file_name;
    }
    if(!label_file.is_open()) {
        throw std::string("can not open file : ") + label_file_name;
    }
    ReadMnistHeader(data_file, label_file, num_data, img_h, img_w);

    size = img_h * img_w;
    data.resize(num_data * size);
    label.resize(num_data);
    for(int n = 0; n < num_data; ++n){
        std::vector<flatbuffers::Offset<mlfe::serializable::TensorBlob>> tbs_std_vec;
        std::string tbs_str;
        data_file.read((char *)data.data() + n * size, size);
        label_file.read((char *)label.data() + n, 1);
        auto tb_data = MakeSerialiableTensor(fbb, "data" + std::to_string(n + 1), data.data() + n * size, { img_h, img_w });
        auto tb_label = MakeSerialiableTensor(fbb, "label" + std::to_string(n + 1), label.data() + n, { 1 });
        tbs_std_vec.push_back(tb_data);
        tbs_std_vec.push_back(tb_label);
        auto tbs_fb_vec = fbb.CreateVector(tbs_std_vec.data(), tbs_std_vec.size());
        auto tbs = mlfe::serializable::CreateTensorBlobs(fbb, tbs_fb_vec);
        fbb.Finish(tbs);
        tbs_str.assign(reinterpret_cast<char *>(fbb.GetBufferPointer()), fbb.GetSize());
        db->Put(std::to_string(n + 1), tbs_str);
        fbb.Clear();
    }
    data_file.close();
    label_file.close();
}

void VerifyDB(std::vector<unsigned char> &data, 
              std::vector<unsigned char> &label, 
              std::shared_ptr<mlfe::DataBase> db
             )
{
    flatbuffers::FlatBufferBuilder fbb;
    auto check_equal = [&data, &label](
        int cur_batch,
        const mlfe::serializable::TensorBlob *data_tb_fb,
        const mlfe::serializable::TensorBlob *label_tb_fb
        ){
        int size = 28 * 28;
        for(int i = 0; i < size; ++i){
            const char a = data.data()[cur_batch * size + i];
            const char b = data_tb_fb->data()->Get(i);
            if(a != b){
                throw std::string("data not matches.");
            }
        }
        {
            const char a = label.data()[cur_batch];
            const char b = label_tb_fb->data()->Get(0);
            if(a != b){
                throw std::string("label not matches.");
            }
        }
    };

    int batch_count = 0;
    db->MoveToFirst();
    do{
        std::string data_val;
        db->Get(data_val);
        auto tbs = mlfe::serializable::GetTensorBlobs(data_val.data());
        check_equal(batch_count, tbs->tensors()->Get(0), tbs->tensors()->Get(1));
        ++batch_count;
    } while(db->MoveToNext());

    if(batch_count != (data.size() / (28 * 28))){
        throw std::string("DB size does not match.");
    }
}

void CreateSimpledbForMnist(std::string train_simpledb_name,
                            std::string test_simpledb_name,
                            std::string train_data_path, 
                            std::string train_label_path, 
                            std::string test_data_path,
                            std::string test_label_path
                           )
{
    std::shared_ptr<mlfe::DataBase> sdb = std::make_shared<mlfe::simpledb::SimpleDB>();
    std::vector<unsigned char> training_data, training_label;
    std::vector<unsigned char> test_data, test_label;

    sdb->option.delete_previous = true;
    sdb->option.binary = true;
    sdb->Open(train_simpledb_name);
    SaveMNIST(
        sdb,
        train_data_path,
        train_label_path,
        training_data,
        training_label
    );
    sdb->Close();

    sdb->Open(test_simpledb_name);
    SaveMNIST(
        sdb,
        test_data_path,
        test_label_path,
        test_data,
        test_label
    );
    sdb->Close();

    sdb->option.delete_previous = false;
    sdb->Open(train_simpledb_name);
    VerifyDB(training_data, training_label, sdb);
    sdb->Close();

    sdb->Open(test_simpledb_name);
    VerifyDB(test_data, test_label, sdb);
    sdb->Close();
}
