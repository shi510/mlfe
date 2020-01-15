#include "simpledb_reader.h"
#include "mlfe/math/blas.h"
#include "mlfe/utils/assert.h"
#include "mlfe/utils/db/simple_db.h"
#include "mlfe/flatbuffers/tensor_blob_fb_generated.h"
#include <vector>
#include <algorithm>

namespace mlfe {

SimpleDBReader::SimpleDBReader(
    std::string path
){
    OpenDB(path);
    bg_worker = std::make_shared<ThreadPool>(1);
    wanna_fill.push(std::vector<std::shared_ptr<std::vector<uint8>>>());
    wanna_fill.front().push_back(std::make_shared<std::vector<uint8>>());
    wanna_fill.front().push_back(std::make_shared<std::vector<uint8>>());
}

SimpleDBReader::~SimpleDBReader() {
    Close();
}

void SimpleDBReader::OpenDB(std::string path) {
    db = std::make_shared<simpledb::SimpleDB>();
    db->option.binary = true;
    db->Open(path);
    db->MoveToFirst();
}

void SimpleDBReader::MoveToFirst(){
    db->MoveToFirst();
}

void SimpleDBReader::Close() {
    bg_worker->Wait(0);
}

void SimpleDBReader::FillBuffer(int batch) {
    if (!wanna_fill.empty()) {
        flatbuffers::FlatBufferBuilder builder;
        auto buffers = wanna_fill.front();
        wanna_fill.pop();
        for (int b = 0; b < batch; ++b) {
            const serializable::TensorBlobs * serialized_tb;
            std::string serialized_data;
            db->Get(serialized_data);
            builder.PushFlatBuffer(reinterpret_cast<const unsigned char *>(serialized_data.data()), serialized_data.size());
            serialized_tb = serializable::GetTensorBlobs(builder.GetBufferPointer());

            int num_data = serialized_tb->tensors()->size();
            for(int t = 0; t < num_data; ++t){
                auto obj = serialized_tb->tensors()->Get(t);
                auto ptr = obj->data()->data();
                int data_size = obj->data()->size();
                int offset = b * data_size;
                if(buffers[t]->size() != batch * data_size){
                    buffers[t]->resize(batch * data_size);
                }
                for(int n = 0; n < data_size; ++n) {
                    auto buffer_ptr = buffers[t]->data();
                    buffer_ptr[offset + n] = ptr[n];
                }
            }
            builder.Clear();
            if (!db->MoveToNext()) {
                db->MoveToFirst();
            }
        }
        wanna_consume.push(buffers);
    }
}
} // end namespace mlfe
