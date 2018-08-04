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

const std::vector<int> &Shape::Dims() const{
    return _dims;
}

void Shape::Clear(){
    _dims.clear();
}

Variable::Variable() : ti(type::float32()){
    _name = std::make_shared<std::string>("Variable");
    _id = std::make_shared<UniqueID>();
    _shape = std::make_shared<class Shape>();
    _size = 0;
}

Variable::Variable(std::string name) : ti(type::float32()){
    _name = std::make_shared<std::string>(name);
    _id = std::make_shared<UniqueID>();
    _shape = std::make_shared<class Shape>();
    _size = 0;
}

Variable::Variable(std::vector<int> shape) : ti(type::float32()){
    _name = std::make_shared<std::string>("Variable");
    _id = std::make_shared<UniqueID>();
    _shape = std::make_shared<class Shape>(shape);
    _size = std::accumulate(_shape->Dims().begin(),
        _shape->Dims().end(), 1, std::multiplies<int>());
}

int Variable::Size() const{
    return std::accumulate(_shape->Dims().begin(),
        _shape->Dims().end(), 1, std::multiplies<int>());
}

int Variable::Dims() const{
    return _shape->Dims().size();
}

int Variable::Dim(int idx) const{
    return _shape->Dims()[idx];
}

std::vector<int> Variable::Shape() const{
    return _shape->Dims();
}

void Variable::Reshape(std::vector<int> shape, type::TypeInfo ti){
    _shape->reshape(shape);
    _size = std::accumulate(_shape->Dims().begin(),
        _shape->Dims().end(), 1, std::multiplies<int>());
    this->ti = ti;
}

type::TypeInfo Variable::Type() const{
    return ti;
}
} // end namespace mlfe;
