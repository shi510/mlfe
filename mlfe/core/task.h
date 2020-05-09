#pragma once
#include <vector>
#include <functional>
#include <tuple>
#include <memory>

namespace mlfe{

class task_impl
{
public:
	virtual void run() = 0;

	virtual ~task_impl(){}
};

template <typename _Fn, typename... _Args>
class task_delegate : public task_impl
{
	typedef std::function<void(_Args...)> _FnType;

	_FnType __fn;
	std::tuple<_Args...> __args;

public:
	task_delegate(_Fn fn, _Args... args)
		:__fn(fn), __args(std::make_tuple(args...)){}

	void run() override
	{
		std::apply(__fn, __args);
	}

	~task_delegate() override{}
};

class task
{
	template <typename _Fn, typename... _Args> friend
		task make_task(_Fn, _Args...);
	typedef std::shared_ptr<task_impl> task_impl_ptr;

	struct pimpl
	{
		// task implementation.
		task_impl_ptr t;
		// dependencies to execute this task.
		std::vector<task> deps;
	};
	std::shared_ptr<pimpl> __pimpl;

public:
	task() : __pimpl(std::make_shared<pimpl>()){}

	void precede(task t)
	{
		t.__pimpl->deps.push_back(*this);
	}

	void run() const
	{
		for(auto& dep : __pimpl->deps)
		{
			dep.run();
		}

		if(__pimpl->t)
		{
			__pimpl->t->run();
		}
	}

	bool empty() const
	{
		return __pimpl->t == nullptr;
	}
};

template <typename _Fn, typename... _Args>
task make_task(_Fn fn, _Args... args)
{
	task t;
	t.__pimpl->t = std::make_shared<task_delegate<_Fn, _Args...>>(fn, args...);
	return t;
}

} // end namespace mlfe
