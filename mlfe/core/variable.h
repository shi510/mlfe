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
    Variable(const bool trainable = false);

    Variable(std::string name, const bool trainable = false);

    Variable(std::vector<int> shape, const std::string name = "", const bool trainable = false);

    Variable(const Variable &) = default;

    std::string name() const;

    void set_name(std::string name);

    int size() const;

    int dims() const;

    int dim(int idx) const;

    std::vector<int> shape() const;

    void reshape(std::vector<int> shape, type::TypeInfo ti = type::float32());

    type::TypeInfo type() const;

    void set_trainable(const bool trainable);

    bool trainable() const;

private:
    std::shared_ptr<std::string> _name;
    std::shared_ptr<class Shape> _shape;
    type::TypeInfo ti;
    int _size;
    bool __trainable;
};
} // end namespace mlfe
#endif // end ifndef __VARIABLE_HPP__
