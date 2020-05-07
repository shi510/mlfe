#pragma once
#include "mlfe/module/callbacks/callback.h"
#include <string>

namespace mlfe{
namespace module{

class reduce_lr : public callback
{
public:
	reduce_lr(std::string monitor,
		int patience = 5,
		double factor = 1e-1,
		double min_lr = 1e-4);

	void on_train_begin(const int epoch,
		const std::map<std::string, float>& logs) override;

private:
	std::string __monitor;
	int __patience;
	double __factor;
	double __min_lr;
	int __count;
	double __best;
};

} // end namespace module
} // end namespace mlfe