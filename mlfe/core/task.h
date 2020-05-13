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

	virtual bool is_mutated() const = 0;

	virtual void set_mutation(bool m) = 0;

	virtual ~task_impl(){}
};

template <typename _Fn, typename... _Args>
class task_delegate : public task_impl
{
	typedef std::function<void(_Args...)> _FnType;

	_FnType __fn;
	std::tuple<_Args...> __args;
	bool __mutation;

public:
	task_delegate(_Fn fn, _Args... args)
		:__fn(fn), __args(std::make_tuple(args...)), __mutation(true){}

	void run() override
	{
		std::apply(__fn, __args);
	}

	bool is_mutated() const override
	{
		return __mutation;
	}

	void set_mutation(bool m) override
	{
		__mutation = m;
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
		task_impl_ptr func;
		// dependencies to execute this task.
		std::vector<task> deps;
		// references.
		std::vector<task> refs;
	};
	std::shared_ptr<pimpl> __pimpl;

public:
	task() : __pimpl(std::make_shared<pimpl>()){}

	void set_task(const task& t)
	{
		__pimpl->func = t.__pimpl->func;
	}

	void precede(task t)
	{
		t.__pimpl->deps.push_back(*this);
		__pimpl->refs.push_back(t);
	}

	void run()
	{
		for(auto& dep : __pimpl->deps)
		{
			dep.run();
		}

		if(__pimpl->func)
		{
			__pimpl->func->run();
		}
	}

	void run_only_mutated()
	{
		for(auto& dep : __pimpl->deps)
		{
			dep.run_only_mutated();
		}

		if(__pimpl->func && this->is_mutated())
		{
			__pimpl->func->run();
			this->set_mutation(false);
			for(auto& r : __pimpl->refs)
			{
				r.set_mutation(true);
			}
		}
	}

	void run_without_dependencies()
	{
		if(__pimpl->func && this->is_mutated())
		{
			__pimpl->func->run();
			this->set_mutation(false);
			for(auto& r : __pimpl->refs)
			{
				r.set_mutation(true);
			}
		}
	}

	bool is_mutated() const
	{
		return __pimpl->func->is_mutated();
	}

	void set_mutation(bool m)
	{
		__pimpl->func->set_mutation(m);
	}

	bool empty() const
	{
		return __pimpl->func == nullptr;
	}
};

template <typename _Fn, typename... _Args>
task make_task(_Fn fn, _Args... args)
{
	task t;
	t.__pimpl->func = std::make_shared<task_delegate<_Fn, _Args...>>(fn, args...);
	return t;
}

} // end namespace mlfe
