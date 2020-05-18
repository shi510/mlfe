#pragma once
#include <string>
#include <memory>
#include <map>

namespace mlfe {	

template <typename RegType>
class registry {
	template <typename, typename>
	friend class registerer;

public:
	registry() = delete;

	static RegType create(const std::string& name);

private:
	static bool regist(const std::string name, RegType rt);

	static std::map<std::string, RegType>& get();
};

template <typename RegType>
std::map<std::string, RegType>& registry<RegType>::get()
{
	static std::map<std::string, RegType> reg;
	return reg;
}

template <typename RegType>
bool registry<RegType>::regist(const std::string name, RegType rt){
	auto& reg = registry<RegType>::get();
	auto it = reg.find(name);
	if(it == reg.end()){
		reg.insert({ name, rt });
		return true;
	}
	return false;
}

template <typename RegType>
RegType registry<RegType>::create(const std::string& name){
	auto& op_reg = registry<RegType>::get();
	auto it = op_reg.find(name);
	if(it != op_reg.end()){
		return it->second;
	}
	return nullptr;
}

//
// end of registry class implementation
//

template <typename Derived, typename Reg>
class registerer {
protected:
	template <typename ...Args>
	static bool regist(const std::string name, Args ...args) {
		return Reg::regist(name, Derived::create(args...));
	}
};

//
// end of registerer class implementation
//

// Example for custom registry.
/*
class base_class
{
public:
	std::string desc;
};

class your_registry
	: public registry<std::shared_ptr<base_class>>{};

template <typename T>
struct your_registerer : public registerer<your_registerer<T>, your_registry>
{
	static std::shared_ptr<base_class> create(std::string desc)
	{
		auto obj_ptr = std::make_shared<T>();
		obj_ptr->desc = desc;
		return obj_ptr;
	}
};

class class_impl : public base_class, public your_registerer<class_impl>
{
private:
	static bool __is_registered;
};

bool class_impl::__is_registered =
	class_impl::regist("name of class_impl", "description of your class_impl");

Then, use as below.

auto obj = your_registry::create("name of class_impl");
std::cout<<obj->desc<<std::endl;
*/

} // end namespace mlfe