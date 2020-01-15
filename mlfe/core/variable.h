#ifndef __VARIABLE_HPP__
#define __VARIABLE_HPP__
#include "mlfe/utils/types.h"
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

    const std::vector<int> &dims() const;

    void clear();

private:
    std::vector<int> _dims;
};

class Variable{
public:
    Variable();

    Variable(std::string name);

    Variable(std::vector<int> shape);

    Variable(const Variable &) = default;

    std::string name() const;

    void set_name(std::string name);

    int size() const;

    int dims() const;

    int dim(int idx) const;

    std::vector<int> shape() const;

    void reshape(std::vector<int> shape, type::TypeInfo ti = type::float32());

    type::TypeInfo type() const;

private:
    std::shared_ptr<std::string> _name;
    std::shared_ptr<class Shape> _shape;
    type::TypeInfo ti;
    int _size;
};
} // end namespace mlfe
#endif // end ifndef __VARIABLE_HPP__
