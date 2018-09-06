#include "cifar.h"
#include <memory>
#include <mlfe/utils/db/simple_db.h>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>
#include <mlfe/utils/db/simpledb_reader.h>
#include <opencv2/opencv.hpp>
#include <numeric>

typedef mlfe::serializable::TensorBlob SerializableTensor;
typedef flatbuffers::FlatBufferBuilder FBB;
typedef flatbuffers::Offset<SerializableTensor> OffsetSerialTensor;

OffsetSerialTensor Serialize(FBB &fbb,
                             std::string name,
                             const uint8_t *data_ptr,
                             std::vector<int> dim
                            )
{
    int size;
    size = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<int>());
    auto name_fb = fbb.CreateString(name);
    auto data_fb = fbb.CreateVector(data_ptr, size * sizeof(uint8_t));
    auto dim_fb = fbb.CreateVector(dim.data(), dim.size());
    auto tb_fb = mlfe::serializable::CreateTensorBlob(fbb, name_fb, data_fb, dim_fb);
    return tb_fb;
}

void VerifyCifar(std::shared_ptr<mlfe::DataBase> db, 
                 std::vector<unsigned char> &data,
                 std::vector<unsigned char> &label
                )
{
    FBB fbb;
    auto check_equal = [&data, &label](int cur_batch,
                                       const SerializableTensor *data_tb_fb,
                                       const SerializableTensor *label_tb_fb
                                      )
    {
        const int size = 32 * 32 * 3;
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

    if(batch_count != (data.size() / (32 * 32 * 3))){
        throw std::string("DB size does not match.");
    }
}

void SaveCifar10(std::shared_ptr<mlfe::DataBase> db,
                 std::vector<std::string> train_batch_names,
                 std::vector<unsigned char> &data,
                 std::vector<unsigned char> &label
                )
{
    const int img_c = 3;
    const int img_h = 32;
    const int img_w = 32;
    const int img_size = img_c * img_h * img_w;
    const int num_batch = train_batch_names.size();
    const int num_data = 10000;
    std::ifstream train_file;
    FBB fbb;

    data.resize(num_batch * num_data * img_size);
    label.resize(num_batch * num_data);
    if(data.size() != num_batch * num_data * img_size){
        throw std::string("Can not allocate memory for data size : ") +
            std::to_string(num_batch * num_data * img_size);
    }
    if(label.size() != num_batch * num_data){
        throw std::string("Can not allocate memory for label size : ") +
            std::to_string(num_batch * num_data);
    }
    for(int n = 0; n < num_batch; ++n){
        int cur_batch = n * num_data;
        train_file.open(train_batch_names[n], std::ios::binary);
        if(!train_file.is_open()){
            throw std::string("can not open file : ") + train_batch_names[n];
        }
        
        for(int i = 0; i < num_data; ++i){
            std::string cur_idx_str = std::to_string(n * num_data + i + 1);
            std::vector<OffsetSerialTensor> tbs_std_vec;
            std::string tbs_str;
            unsigned char *data_ptr = data.data() + cur_batch * img_size + i * img_size;
            unsigned char *label_ptr = label.data() + cur_batch + i;
            train_file.read((char *)label_ptr, 1);
            train_file.read((char *)data_ptr, img_size);
            auto tb_data = Serialize(fbb, "data" + cur_idx_str, data_ptr, { img_c, img_h, img_w });
            auto tb_label = Serialize(fbb, "label" + cur_idx_str, label_ptr, { 1 });
            tbs_std_vec.push_back(tb_data);
            tbs_std_vec.push_back(tb_label);
            auto tbs_fb_vec = fbb.CreateVector(tbs_std_vec.data(), tbs_std_vec.size());
            auto tbs = mlfe::serializable::CreateTensorBlobs(fbb, tbs_fb_vec);
            fbb.Finish(tbs);
            tbs_str.assign(reinterpret_cast<char *>(fbb.GetBufferPointer()), fbb.GetSize());
            db->Put(cur_idx_str, tbs_str);
            fbb.Clear();
        }
        train_file.close();
    }
}

void CreateSimpledbForCifar(std::string train_simpledb_name,
                            std::string test_simpledb_name,
                            std::vector<std::string> train_batch_paths, 
                            std::string test_batch_path
                           )
{
    std::shared_ptr<mlfe::DataBase> sdb = std::make_shared<mlfe::simpledb::SimpleDB>();
    std::vector<unsigned char> train_data, train_label;
    std::vector<unsigned char> test_data, test_label;

    sdb->option.delete_previous = true;
    sdb->option.binary = true;

    //read cifar train data and save to simpledb.
    std::cout << "Writing cifar10 train data...";
    sdb->Open(train_simpledb_name);
    SaveCifar10(sdb, train_batch_paths, train_data, train_label);
    sdb->Close();
    std::cout << "  Done." << std::endl;

    //read cifar test data and save to simpledb.
    std::cout << "Writing cifar10 test data...";
    sdb->Open(test_simpledb_name);
    SaveCifar10(sdb, { test_batch_path }, test_data, test_label);
    sdb->Close();
    std::cout << "  Done." << std::endl;

    // verify train data.
    std::cout << "Verifing cifar10 train data...";
    sdb->option.delete_previous = false;
    sdb->Open(train_simpledb_name);
    VerifyCifar(sdb, train_data, train_label);
    sdb->Close();
    std::cout << "  Done." << std::endl;

    // verify test data.
    std::cout << "Verifing cifar10 test data...";
    sdb->Open(test_simpledb_name);
    VerifyCifar(sdb, test_data, test_label);
    sdb->Close();
    std::cout << "  Done." << std::endl;
}