#include "simple_db.hpp"
#include "../assert.hpp"
#include <sstream>
#include <vector>

namespace mlfe{ namespace simpledb{
    
SimpleDB::SimpleDB(){
    header_info.signature = 0x57957295;
    header_info.version = 0x00000001;
    header_info.number_of_items = 0;
    header_info.disk_info_offset = sizeof(header_info);
}

SimpleDB::~SimpleDB(){
    Close();
}

void SimpleDB::Close() {
    if(FileIO::IsOpen()){
        WriteHeader();
        WriteDiskInfo();
        FileIO::Close();
    }
}

void SimpleDB::Open(const std::string name) {
    try{
        FileIO::Open(name, std::ios::in | std::ios::out | std::ios::binary);
    }
    catch(std::string &e){
        FileIO::Open(name, std::ios::out | std::ios::binary);
        FileIO::Close();
        FileIO::Open(name, std::ios::in | std::ios::out | std::ios::binary);
    }
    SeekToEnd();
    file_size = GetPosition();
    if(file_size < HeaderSize() && file_size != 0){
        throw std::string("Header size does not match.");
    }
    if(file_size == 0){
        WriteHeader();
    }
    else{
        ReadHeader();
        ReadDiskInfo();
    }
    last_cursor = header_info.disk_info_offset;
}

void SimpleDB::Get(const std::string key, std::string &val) {
    runtime_assert(disk_info_tree.count(key) == 1, "Key not exists.");
    ItemDiskInfo target= disk_info_tree[key];
    val.resize(target.data_size);
    SeekFromFirstTo(target.pos_data_from_first);
    Read(const_cast<char *>(val.c_str()), val.size());
}

void SimpleDB::Put(const std::string key, const std::string val) {
    runtime_assert(disk_info_tree.count(key) == 0, "Key already exists.");
    ItemDiskInfo new_data;
    new_data.data_size = val.size();
    new_data.pos_data_from_first = last_cursor;
    disk_info_tree[key] = new_data;
    Write(const_cast<char *>(val.c_str()), val.size());
    last_cursor += new_data.data_size;
}

void SimpleDB::Delete(const std::string key) {
    throw std::string("SimpleDB Delete() function dose not supported yet.");
}

int SimpleDB::NumData() {
    return disk_info_tree.size();
}

void SimpleDB::ReadHeader(){
    HeaderInfo temp;
    SeekToFirst();
    Read(reinterpret_cast<char *>(&temp), sizeof(temp));
    if (temp.signature != header_info.signature) {
        std::ostringstream info;
        info<<"Signature not matches.";
        info<<"[";
        info<<temp.signature;
        info<<"vs";
        info<<header_info.signature;
        info<<"]";
        throw info.str();
    }
    if (temp.version != header_info.version) {
        std::ostringstream info;
        info << "Version not matches.";
        info << "[";
        info << temp.version << "(opened file)";
        info << "vs";
        info << header_info.version << "(current library)";
        info << "]";
        throw info.str();
    }
    header_info.number_of_items = temp.number_of_items;
    header_info.disk_info_offset = temp.disk_info_offset;
}

void SimpleDB::WriteHeader(){
    SeekToFirst();
    header_info.version = header_info.version;
    header_info.signature = header_info.signature;
    header_info.number_of_items = disk_info_tree.size();
    header_info.disk_info_offset = HeaderSize() + GetAllItemSize();
    Write(reinterpret_cast<char *>(&header_info), sizeof(header_info));
}

void SimpleDB::ReadDiskInfo(){
    std::vector<ItemDiskInfo> vec_disk_info;
    std::vector<char> keys;
    uint32_t total_key_size = 0;
    uint32_t accum_key_size = 0;
    uint32_t key_count = 0;
    
    vec_disk_info.resize(header_info.number_of_items);
    SeekFromFirstTo(header_info.disk_info_offset);
    Read(
         reinterpret_cast<char *>(vec_disk_info.data()),
         sizeof(ItemDiskInfo) * header_info.number_of_items
         );
    
    total_key_size = file_size - (GetPosition());
    keys.resize(total_key_size);
    Read(keys.data(), total_key_size);
    
    key_count = GetCountOfRead();
    runtime_assert(
                   key_count == total_key_size,
                   "the number of data and the number of key are not same. (maybe, file has been corrupted.)"
                   );
    for(int n = 0; n < header_info.number_of_items; ++n){
        std::string key(keys.data() + accum_key_size);
        vec_disk_info[n].data_size = vec_disk_info[n].data_size;
        vec_disk_info[n].pos_data_from_first = vec_disk_info[n].pos_data_from_first;
        disk_info_tree[key] = vec_disk_info[n];
        accum_key_size += key.size() + 1;
    }
    SeekFromFirstTo(HeaderSize() + GetAllItemSize());
}

void SimpleDB::WriteDiskInfo(){
    SeekFromFirstTo(header_info.disk_info_offset);
    /*
     * TODO:
     * Merge ItemDiskInfo and key into one string for one write op.
     * maybe it also needs the algorithm that not use for-loop.
     */
    for(auto &iter : disk_info_tree){
        iter.second.data_size = iter.second.data_size;
        iter.second.pos_data_from_first = iter.second.pos_data_from_first;
        Write(reinterpret_cast<char *>(&iter.second), sizeof(ItemDiskInfo));
    }
    
    for(auto &iter : disk_info_tree){
        Write(const_cast<char *>(iter.first.c_str()), iter.first.size() + 1);
    }
}

int SimpleDB::HeaderSize(){
    return sizeof(HeaderInfo);
}

uint32_t SimpleDB::GetAllItemSize(){
    uint32_t size = 0;
    /*
     * TODO:
     * implement the faster method than the accumulation one by for-loop.
     */
    for(auto &item : disk_info_tree){
        size += item.second.data_size;
    }
    return size;
}
    
} /* namespace simpledb */
} /* namespace mlfe */
