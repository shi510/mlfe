#ifndef __FILE_IO_HPP__
#define __FILE_IO_HPP__
#include <string>
#include <fstream>

namespace mlfe{

class FileIO {
public:
    virtual ~FileIO() {
        Close();
    }

    void Open(std::string name, std::ios::openmode mode) {
		if (file.is_open()) {
			throw std::string("File is already opened.");
		}
		file.open(name, mode);
        if(!file.is_open()){
            throw std::string("File can not open. - ") + name;
        }
	}

	void Close() {
		if (file.is_open()) {
			file.close();
		}
	}
    
    bool Exists(std::string name){
        std::ifstream f(name);
        if(f.is_open()){
            f.close();
            return true;
        }
        else{
            return false;
        }
    }
    
    bool CreateFile(std::string name){
        std::ofstream f(name);
        if(f.is_open()){
            f.close();
            return true;
        }
        else{
            return false;
        }
    }

    void Read(char *ptr, int size){
        file.read(ptr, size);
    }
    
    int GetCountOfRead(){
        return file.gcount();
    }

    void Write(char *ptr, int size){
        file.write(ptr, size);
    }
    
    void SeekFromFirstTo(int pos){
        file.seekg(std::ios::beg + pos);
    }
    
    void SeekFromEndTo(int pos){
        file.seekg(std::ios::end - pos);
    }
    
    void SeekToFirst(){
        file.seekg(std::ios::beg);
    }
    
    void SeekToEnd(){
        file.seekp(std::ios::beg, std::ios::end);
    }
    
    int GetPosition(){
        return file.tellg();
    }

    bool IsOpen(){
        return file.is_open();
    }
    
    void Flush(){
        file.flush();
    }

private:
	std::fstream file;
};

} /* namespace mlfe */
#endif /* __FILE_IO_HPP__ */
