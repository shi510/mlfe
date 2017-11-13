#ifndef __SIMPLE_DB_HPP__
#define __SIMPLE_DB_HPP__
#include "data_base.hpp"
#include <map>

namespace mlfe{ namespace simpledb{

/*
 * TODO:
 * more algorithms need: B-Tree, data cashing, reducing direct calling of IO seek,
 * async algorithm for put, get, delete,
 * data compression to reduce DB size,
 * efficient key-value management,
 * db options, db status.
 */
class SimpleDB final : public DataBase{
public:
    SimpleDB();
    
    ~SimpleDB();
    
    void Close() override;
    
    void Open(const std::string name) override;
    
    void Get(const std::string key, std::string &val) override;
    
    void Put(const std::string key, const std::string val) override;
    
    void Delete(const std::string key) override;
    
    int NumData() override;
    
protected:
    void ReadHeader();
    
    void WriteHeader();
    
    void ReadDiskInfo();
    
    void WriteDiskInfo();
    
    int HeaderSize();
    
    uint32_t GetAllItemSize();
    
private:
    /*
     * TODO:
     * Define max item, max size of data, max size of key.
     * dealing with Padding bit, when use int64_t in odd.
     */
    struct HeaderInfo{
        uint32_t signature;
        uint32_t version;
        uint32_t number_of_items;
        uint32_t disk_info_offset;
    } header_info;
    
    struct ItemDiskInfo{
        uint32_t data_size;
        uint32_t pos_data_from_first;
    };
    
    /*
     * TODO:
     * Define max tree-level.
     * Define function to transform from tree to string.
     */
    std::map<std::string, ItemDiskInfo> disk_info_tree;
    uint32_t file_size;
    uint32_t last_cursor;
};

} /* namespace simpledb */
} /* namespace mlfe */
#endif /* __SIMPLE_DB_HPP__ */
