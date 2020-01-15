#ifndef __SIMPLEDB_READER_HPP__
#define __SIMPLEDB_READER_HPP__

#include "mlfe/core/tensor.h"
#include "mlfe/utils/thread_pool.h"
#include "mlfe/utils/db/data_base.h"

namespace mlfe {

class SimpleDBReader{
template <class T>
using Ptr = std::shared_ptr<T>;
template <class T>
using Vec = std::vector<T>;
template <class T>
using RefWrapVec = std::reference_wrapper<Vec<T>>;
using uint8 = type::uint8::T;
public:
    SimpleDBReader(std::string path);

    ~SimpleDBReader();

    template <class T>
    void Read(int batch, std::vector<RefWrapVec<T>> tensors);

    void MoveToFirst();

    void Close();

protected:
    void OpenDB(std::string path);

    void FillBuffer(int batch);

private:
    std::shared_ptr<ThreadPool> bg_worker;
    std::queue<Vec<Ptr<std::vector<uint8>>>> wanna_consume;
    std::queue<Vec<Ptr<std::vector<uint8>>>> wanna_fill;
    std::shared_ptr<DataBase> db;
};

template <class T>
void SimpleDBReader::Read(int batch, std::vector<RefWrapVec<T>> tensors) {
    bg_worker->Wait(0);
    if(!wanna_fill.empty()){
        bg_worker->AddTask(std::bind(&SimpleDBReader::FillBuffer, this, batch), 0);
        bg_worker->Wait(0);
    }
    auto buffer = wanna_consume.front();
    wanna_consume.pop();
    std::copy(
        buffer[0]->data(),
        buffer[0]->data() + buffer[0]->size(),
        tensors[0].get().begin()
    );
    std::copy(
        buffer[1]->data(),
        buffer[1]->data() + buffer[1]->size(),
        tensors[1].get().begin()
    );
    wanna_fill.push(buffer);
    bg_worker->AddTask(std::bind(&SimpleDBReader::FillBuffer, this, batch), 0);
}

} // end namespace mlfe
#endif // end ifndef __SIMPLEDB_READER_HPP__
