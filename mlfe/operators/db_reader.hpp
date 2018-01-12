#ifndef __DB_READER_OP_HPP__
#define __DB_READER_OP_HPP__
#include <queue>
#include <thread>
#include "operator.hpp"
#include "../utils/thread_pool.hpp"
#include "../utils/db/data_base.hpp"
#include "../utils/db/simple_db.hpp"
#include "../utils/assert.hpp"
#include "../flatbuffers/tensor_blob_fb_generated.h"

namespace mlfe{

template <class DataType, class DeviceContext>
class DBReaderOp final : public Operator<DeviceContext>{
public:
    explicit DBReaderOp(
                        OperatorIO &opio,
                        ItemHolder *ih
                        );
    
    ~DBReaderOp();
    
    void Compute() override;
    
protected:
    void OpenDB(std::string path, std::string type){
        if(!type.compare("SimpleDB")){
            db = std::make_shared<simpledb::SimpleDB>();
        }
        else{
            throw std::string("[DB Reader] Does not support -> ") + type;
        }
        db->option.binary = true;
        db->Open(path);
        db->MoveToFirst();
    }
    
    void FillBuffer(){
        if(!wanna_fill.empty()){
            flatbuffers::FlatBufferBuilder builder;
            auto tbs = wanna_fill.front();
            wanna_fill.pop();
            for(int b = 0; b < batch_size; ++b){
                const serializable::TensorBlobs * serialized_tb;
                std::string serialized_data;
                unsigned int data_size = (tbs[0]->Size() / tbs[0]->Dim(0));
                db->Get(serialized_data);
                builder.PushFlatBuffer(reinterpret_cast<const unsigned char *>(serialized_data.data()), serialized_data.size());
                serialized_tb = serializable::GetTensorBlobs(builder.GetBufferPointer());
                
                tbs[0]->CopyToDevice(
                                     b * data_size,
                                     data_size,
                                     static_cast<const unsigned char *>(serialized_tb->tensors()->Get(0)->data()->data())
                                     );
                
                if(has_label){
                    data_size = 1;
                    
                    tbs[1]->CopyToDevice(
                                         b * data_size,
                                         data_size,
                                         static_cast<const unsigned char *>(serialized_tb->tensors()->Get(1)->data()->data())
                                         );
                }
                builder.Clear();
                if(!db->MoveToNext()){
                    db->MoveToFirst();
                }
            }
            wanna_consume.push(tbs);
        }
    }
    
private:
    enum OutputSchema{y, label};
    int batch_size;
    bool has_label;
    ThreadPool background_worker;
    std::queue<std::vector<std::shared_ptr<TensorBlob<DeviceContext>>>> wanna_consume;
    std::queue<std::vector<std::shared_ptr<TensorBlob<DeviceContext>>>> wanna_fill;
    std::shared_ptr<DataBase> db;
};

} /* namespace mlfe */
#endif /* __DB_READER_OP_HPP__ */
