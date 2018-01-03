#ifndef __DB_READER_OP_HPP__
#define __DB_READER_OP_HPP__
#include <vector>
#include <queue>
#include <thread>
#include "operator.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "../utils/assert.hpp"
#include "../utils/db/simple_db.hpp"
#include "../utils/thread_pool.hpp"
#include "../flatbuffers/tensor_blob_fb_generated.h"

namespace mlfe{

template <class DataType, class DeviceContext>
class DBReaderOp final : public Operator<DeviceContext>{
public:
    explicit DBReaderOp(
                        std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                        ParamDef param = ParamDef()
                        ) : Operator<DeviceContext>(std::vector<std::shared_ptr<TensorBlob<DeviceContext>>>(), outputs, param), background_worker(1) {
        std::string db_path = "";
        std::string db_type = "";
        std::vector<int> data_dim, label_dim;
        std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> tbs;
        std::shared_ptr<TensorBlob<DeviceContext>> buffer_data, buffer_label;
        batch_size = 0;
        flatten = false;
        has_label = false;
        this->GetParam().GetParamByName("DataBasePath", db_path);
        this->GetParam().GetParamByName("DataBaseType", db_type);
        this->GetParam().GetParamByName("BatchSize", batch_size);
        this->GetParam().GetParamByName("Flatten", flatten);
        this->GetParam().GetParamByName("HasLabel", has_label);
        this->GetParam().GetParamByName("DataShape", data_dim);
        this->GetParam().GetParamByName("LabelShape", label_dim);
        if(db_path.empty() && db_type.empty()){
            throw std::string("You must feed \"DataBasePath\" Parameter and \"DataBaseType\" Parameter.");
        }
        if(has_label){
            runtime_assert(this->Outputs() == 2, "The number of outputs must be 2(data, label).");
        }
        else{
            runtime_assert(this->Outputs() == 1, "The number of outputs must be 1(data).");
        }
        runtime_assert(batch_size > 0, "The batch size must be greater than 0.");
        
        OpenDB(db_path, db_type);
        buffer_data = std::make_shared<TensorBlob<DeviceContext>>();
        buffer_label = std::make_shared<TensorBlob<DeviceContext>>();
        this->Output(0)->template Reshape<DataType>(data_dim);
        this->Output(1)->template Reshape<DataType>(label_dim);
        buffer_data->template ReshapeLike<DataType>(this->Output(0));
        buffer_label->template ReshapeLike<DataType>(this->Output(1));
        tbs.push_back(buffer_data);
        tbs.push_back(buffer_label);
        wanna_fill.push(tbs);
        background_worker.AddTask(std::bind(&DBReaderOp::FillBuffer, this), 0);
    }
    
    ~DBReaderOp(){
        if(db->IsOpen()){
            db->Close();
        }
        background_worker.Wait(0);
    }
    
    void Compute() override {
        if(wanna_consume.empty()){
            if(!background_worker.IsFinished(0)){
                background_worker.Wait(0);
            }
        }
        
        auto vec_of_tb = wanna_consume.front();
        wanna_consume.pop();
        this->Output(0)->CopyToDevice(
                                      0,
                                      vec_of_tb[0]->Size(),
                                      vec_of_tb[0]->template GetPtrConst<DataType>()
                                      );

        if(has_label){
            this->Output(1)->CopyToDevice(
                                          0,
                                          vec_of_tb[1]->Size(),
                                          vec_of_tb[1]->template GetPtrConst<DataType>()
                                          );
        }
        
        wanna_fill.push(vec_of_tb);
        background_worker.AddTask(std::bind(&DBReaderOp::FillBuffer, this), 0);
    }
    
protected:
    void OpenDB(std::string path, std::string type){
        if(!type.compare("SimpleDB")){
            db = std::make_shared<simpledb::SimpleDB>();
        }
        else{
            throw std::string("Database type dose not match. (current version of db reader operator supports only simpledb.)");
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
                                     static_cast<const DataType *>(serialized_tb->tensors()->Get(0)->data()->data())
                                     );
                
                if(has_label){
                    data_size = 1;
                    
                    tbs[1]->CopyToDevice(
                                         b * data_size,
                                         data_size,
                                         static_cast<const DataType *>(serialized_tb->tensors()->Get(1)->data()->data())
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
    bool flatten;
    bool has_label;
    ThreadPool background_worker;
    std::queue<std::vector<std::shared_ptr<TensorBlob<DeviceContext>>>> wanna_consume;
    std::queue<std::vector<std::shared_ptr<TensorBlob<DeviceContext>>>> wanna_fill;
    std::shared_ptr<DataBase> db;
};

} /* namespace mlfe */
#endif /* __DB_READER_OP_HPP__ */
