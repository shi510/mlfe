#include "mlfe/module/callbacks/callback.h"
#include "mlfe/module/model.h"

namespace mlfe{
namespace module{

void callback::on_epoch_end(const int epoch,
	const std::map<std::string, float>& logs){}

void callback::on_train_begin(const int epoch,
	const std::map<std::string, float>& logs){}

void callback::on_train_end(const int epoch,
	const std::map<std::string, float>& logs){}

void callback::on_test_begin(const int epoch,
	const std::map<std::string, float>& logs){}

void callback::on_test_end(const int epoch,
	const std::map<std::string, float>& logs){}

void callback::set_model(model* m)
{
	__m = m;
}

} // end namespace module
} // end namespace mlfe