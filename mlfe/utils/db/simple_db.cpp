#include "simple_db.hpp"
#include "../assert.hpp"
#include <sstream>
#include <vector>
#include <iostream>

namespace mlfe{ namespace simpledb{
    
SimpleDB::SimpleDB(){
    Init();
}

SimpleDB::~SimpleDB(){
    Close();
}

void SimpleDB::Close() {
    if(FileIO::IsOpen()){
        WriteHeader();
        WriteDiskInfo();
        FileIO::Close();
        Init();
    }
}

void SimpleDB::Open(const std::string name) {
    std::ios::openmode mode = GetFileModeByOption();
    /*
     * TODO:
     * simplify the file open procedure.
     */
    try{
        if(option.create){
            FileIO::CreateFile(name);
        }
        FileIO::Open(name, mode);
    }
    catch(std::string &e){
        throw e;
    }
    SeekToEnd();
    file_size = GetPosition();
    if(file_size < HeaderSize() && file_size != 0){
        throw std::string("Header size does not match.");
    }
    if(file_size == 0 || option.delete_previous){
        WriteHeader();
    }
    else if(!option.create){
        ReadHeader();
        ReadDiskInfo();
    }
    last_cursor = header_info.disk_info_offset;
    SeekFromFirstTo(last_cursor);
}

void SimpleDB::Get(std::string &val) {
    ItemDiskInfo target;
    if(option.ordered){
        target = iter->second;
    }
    else{
        target = *insertion_order_iter->second;
    }
    val.resize(target.data_size);
    SeekFromFirstTo(target.pos_data_from_first);
    Read(const_cast<char *>(val.c_str()), val.size());
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
    insertion_order.push_back(std::make_pair(key, &disk_info_tree[key]));
    Write(const_cast<char *>(val.c_str()), val.size());
    last_cursor += new_data.data_size;
}

void SimpleDB::MoveToFirst(){
    iter = disk_info_tree.begin();
    insertion_order_iter = insertion_order.begin();
}

bool SimpleDB::MoveToNext(){
    ++iter;
    ++insertion_order_iter;
    if(iter == disk_info_tree.end()){
        return false;
    }
    return true;
}

void SimpleDB::Delete(const std::string key) {
    throw std::string("SimpleDB Delete() function dose not supported yet.");
}

int SimpleDB::NumData() {
    return disk_info_tree.size();
}
    
void SimpleDB::Init(){
    header_info.signature = 0x57957295;
    header_info.version = 0x00000001;
    header_info.number_of_items = 0;
    header_info.disk_info_offset = sizeof(header_info);
    disk_info_tree.clear();
    insertion_order.clear();
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
        insertion_order.push_back(std::make_pair(key, &disk_info_tree[key]));
        accum_key_size += key.size() + 1;
    }
    SeekFromFirstTo(HeaderSize() + GetAllItemSize());
}

void SimpleDB::WriteDiskInfo(){
    SeekFromFirstTo(header_info.disk_info_offset);
    
    /*
     * TODO:
     * 1. Merge ItemDiskInfo and key into one string for one write op.
     * maybe it also needs the algorithm that not use for-loop.
     * 2. remove the conditional branch.
     */
    if(option.ordered){
        for(auto &iter : disk_info_tree){
            iter.second.data_size = iter.second.data_size;
            iter.second.pos_data_from_first = iter.second.pos_data_from_first;
            Write(reinterpret_cast<char *>(&iter.second), sizeof(ItemDiskInfo));
        }
        
        for(auto &iter : disk_info_tree){
            Write(const_cast<char *>(iter.first.c_str()), iter.first.size() + 1);
        }
    }
    else{
        for(auto &iter : insertion_order){
            iter.second->data_size = iter.second->data_size;
            iter.second->pos_data_from_first = iter.second->pos_data_from_first;
            Write(reinterpret_cast<char *>(iter.second), sizeof(ItemDiskInfo));
        }
        
        for(auto &iter : insertion_order){
            Write(const_cast<char *>(iter.first.c_str()), iter.first.size() + 1);
        }
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

std::ios::openmode SimpleDB::GetFileModeByOption(){
    std::ios::openmode mode;
    mode = std::ios::in | std::ios::out;
    if(option.binary){
        mode |= std::ios::binary;
    }
    if(option.delete_previous){
        mode |= std::ios::trunc;
    }
    if(!option.binary){
        mode ^= std::ios::binary;
    }
    return mode;
}

} /* namespace simpledb */
} /* namespace mlfe */
