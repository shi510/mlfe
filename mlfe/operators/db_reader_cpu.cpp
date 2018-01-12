#include "db_reader.hpp"
#include "../utils/db/simple_db.hpp"
#include "../utils/assert.hpp"
#include "../flatbuffers/tensor_blob_fb_generated.h"

namespace mlfe{

template <>
DBReaderOp<unsigned char, CPUContext>
::DBReaderOp(
             OperatorIO &opio,
             ItemHolder *ih
             ) : Operator<CPUContext>(opio, ih), background_worker(1){
    std::string db_path, db_type;
    std::vector<int> data_dim, label_dim;
    std::vector<std::shared_ptr<TensorBlob<CPUContext>>> tbs;
    std::shared_ptr<TensorBlob<CPUContext>> buffer_data, buffer_label;
    has_label = false;
    
    runtime_assert(opio.param.HasParam("DatabasePath"),
                   "[DB Reader Op] Not found DatabasePath param.");
    runtime_assert(opio.param.HasParam("DatabaseType"),
                   "[DB Reader Op] Not found DatabaseType param.");
    runtime_assert(opio.param.HasParam("DataShape"),
                   "[DB Reader Op] Not found DataShape param.");
    
    db_path = opio.param.GetParam<std::string>("DatabasePath");
    db_type = opio.param.GetParam<std::string>("DatabaseType");
    data_dim = opio.param.GetParam<std::vector<int>>("DataShape");
    
    if(opio.param.HasParam("HasLabel")){
        runtime_assert(Outputs() == 2,
                       "[DB Reader Op] Outputs() == 2");
        runtime_assert(opio.param.HasParam("LabelShape"),
                       "[DB Reader Op] Not found LabelShape param.");
        has_label = opio.param.GetParam<bool>("HasLabel");
        label_dim = opio.param.GetParam<std::vector<int>>("LabelShape");
    }
    else{
        runtime_assert(Outputs() == 1,
                       "[DB Reader Op] Outputs() == 1");
    }
    batch_size = data_dim[0];
    
    OpenDB(db_path, db_type);
    buffer_data = std::make_shared<TensorBlob<CPUContext>>();
    buffer_label = std::make_shared<TensorBlob<CPUContext>>();
    outputs[0]->Resize<unsigned char>(data_dim);
    outputs[1]->Resize<unsigned char>(label_dim);
    buffer_data->Resize<unsigned char>(*outputs[0]);
    buffer_label->Resize<unsigned char>(*outputs[1]);
    tbs.push_back(buffer_data);
    tbs.push_back(buffer_label);
    wanna_fill.push(tbs);
    background_worker.AddTask(std::bind(&DBReaderOp<unsigned char, CPUContext>::FillBuffer, this), 0);
}

template <>
DBReaderOp<unsigned char, CPUContext>::~DBReaderOp(){
    if(db->IsOpen()){
        db->Close();
    }
    background_worker.Wait(0);
}

template <>
void DBReaderOp<unsigned char, CPUContext>::Compute(){
    if(wanna_consume.empty()){
        if(!background_worker.IsFinished(0)){
            background_worker.Wait(0);
        }
    }
    
    auto vec_of_tb = wanna_consume.front();
    wanna_consume.pop();
    outputs[0]->CopyToDevice(
                             0,
                             vec_of_tb[0]->Size(),
                             vec_of_tb[0]->GetPtrConst<unsigned char>()
                             );
    
    if(has_label){
        outputs[1]->CopyToDevice(
                                 0,
                                 vec_of_tb[1]->Size(),
                                 vec_of_tb[1]->GetPtrConst<unsigned char>()
                                 );
    }
    
    wanna_fill.push(vec_of_tb);
    background_worker.AddTask(std::bind(&DBReaderOp::FillBuffer, this), 0);
}

REGIST_OPERATOR_CPU(DBReader, DBReaderOp<unsigned char, CPUContext>)

} /* namespace mlfe */
