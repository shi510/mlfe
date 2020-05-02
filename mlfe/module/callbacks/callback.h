#pragma once
#include <map>
#include <string>

namespace mlfe{
namespace module{

class model;

class callback
{
public:
	virtual void on_epoch_end(const int epoch,
		const std::map<std::string, float>& logs);

	virtual void on_train_begin(const int epoch,
		const std::map<std::string, float>& logs);

	virtual void on_train_end(const int epoch,
		const std::map<std::string, float>& logs);

	virtual void on_test_begin(const int epoch,
		const std::map<std::string, float>& logs);

	virtual void on_test_end(const int epoch,
		const std::map<std::string, float>& logs);

	void set_model(model* m);

	std::string __name;

protected:
	model* __m;
};

} // end namespace module
} // end namespace mlfe