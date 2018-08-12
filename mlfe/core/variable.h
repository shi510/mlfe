#ifndef __VARIABLE_HPP__
#define __VARIABLE_HPP__
#include "../utils/types.h"
#include <memory>
#include <string>
#include <vector>

namespace mlfe{

class Shape final{
public:
    Shape();

    Shape(std::initializer_list<int> dims);

    Shape(std::vector<int> dims);

    void reshape(std::vector<int> dims);

    const std::vector<int> &Dims() const;

    void Clear();

private:
    std::vector<int> _dims;
};

class Variable{
public:
    Variable();

    Variable(std::string name);

    Variable(std::vector<int> shape);

    Variable(const Variable &) = default;

    std::string Name() const;

    void SetName(std::string name){
        *_name = name;
    }

    int Size() const;

    int Dims() const;

    int Dim(int idx) const;

    std::vector<int> Shape() const;

    void Reshape(std::vector<int> shape, type::TypeInfo ti = type::float32());

    type::TypeInfo Type() const;

private:
    class UniqueID{
    public:
        UniqueID(){
            _id = _next_gen;
            _next_gen += 1;
        }

        unsigned int Id(){
            return _id;
        }
    private:
        unsigned int _id;
        static unsigned int _next_gen;
    };

    std::shared_ptr<UniqueID> _id;
    std::shared_ptr<std::string> _name;
    std::shared_ptr<class Shape> _shape;
    type::TypeInfo ti;
    int _size;
};
} // end namespace mlfe
#endif // end ifndef __VARIABLE_HPP__