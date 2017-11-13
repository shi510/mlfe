#ifndef __DATA_BASE_HPP__
#define __DATA_BASE_HPP__
#include <string>
#include <fstream>
#include "../file_io.hpp"

namespace mlfe{

class DataBase : public FileIO{
public:
    virtual ~DataBase() { }

    virtual void Close() = 0;

    virtual void Open(const std::string name) = 0;

    virtual void Get(const std::string key, std::string &val) = 0;

    virtual void Put(const std::string key, const std::string val) = 0;
    
    virtual void Delete(const std::string key) = 0;

    virtual int NumData() = 0;
};

} /* namespace mlfe */
#endif /* __DATA_BASE_HPP__ */
