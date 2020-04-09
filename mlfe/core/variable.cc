#include "variable.h"
#include <numeric>

namespace mlfe{

Shape::Shape(){}

Shape::Shape(std::initializer_list<int> dims){
    _dims = dims;
}

Shape::Shape(std::vector<int> dims){
    _dims = dims;
}

void Shape::reshape(std::vector<int> dims){
    _dims = dims;
}

const std::vector<int> &Shape::dims() const{
    return _dims;
}

void Shape::clear(){
    _dims.clear();
}

Variable::Variable(const bool trainable)
    : ti(type::float32()), __trainable(trainable)
{
    _name = std::make_shared<std::string>("Variable");
    _shape = std::make_shared<class Shape>();
    _size = 0;
}

Variable::Variable(std::string name, const bool trainable)
    : ti(type::float32()), __trainable(trainable)
{
    _name = std::make_shared<std::string>(name);
    _shape = std::make_shared<class Shape>();
    _size = 0;
}

Variable::Variable(std::vector<int> shape, const std::string name, const bool trainable)
    : ti(type::float32()), __trainable(trainable)
{
    _name = std::make_shared<std::string>(name);
    _shape = std::make_shared<class Shape>(shape);
    _size = std::accumulate(_shape->dims().begin(),
        _shape->dims().end(), 1, std::multiplies<int>());
}

std::string Variable::name() const{
    return *_name;
}

void Variable::set_name(std::string name){
    *_name = name;
}

int Variable::size() const{
    //std::cout << _size << std::endl;
    //std::cout << std::accumulate(_shape->Dims().begin(),
    //    _shape->Dims().end(), 1, std::multiplies<int>()) << std::endl;
    return std::accumulate(_shape->dims().begin(),
        _shape->dims().end(), 1, std::multiplies<int>());
}

int Variable::dims() const{
    return _shape->dims().size();
}

int Variable::dim(int idx) const{
    return _shape->dims()[idx];
}

std::vector<int> Variable::shape() const{
    return _shape->dims();
}

void Variable::reshape(std::vector<int> shape, type::TypeInfo ti){
    _shape->reshape(shape);
    _size = std::accumulate(_shape->dims().begin(),
        _shape->dims().end(), 1, std::multiplies<int>());
    this->ti = ti;
}

type::TypeInfo Variable::type() const{
    return ti;
}

void Variable::set_trainable(const bool trainable)
{
    __trainable = trainable;
}

bool Variable::trainable() const
{
    return __trainable;
}

} // end namespace mlfe;
