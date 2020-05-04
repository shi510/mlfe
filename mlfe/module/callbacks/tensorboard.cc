#include "mlfe/module/callbacks/tensorboard.h"
#include "mlfe/module/model.h"
#include <limits>

namespace mlfe{
namespace module{

tensorboard::tensorboard(std::string log_dir)
{
	__log_dir = log_dir;
}

void tensorboard::on_epoch_end(const int epoch,
	const std::map<std::string, float>& logs)
{
	for (auto& val : logs)
	{
		__cur_writer->add_scalar(val.first, epoch, val.second);
	}
}

void tensorboard::on_train_begin(const int epoch,
	const std::map<std::string, float>& logs)
{
	auto it = __writers.find("train");
	if(it == __writers.end())
	{
		std::string file_path = __log_dir + "/train/tfevents.pb";
		auto writer = std::make_shared<util::summary_writer>(file_path);
		__writers["train"] = writer;
		__cur_writer = writer;
	}
	else
	{
		__cur_writer = it->second;
	}
}

void tensorboard::on_test_begin(const int epoch,
	const std::map<std::string, float>& logs)
{
	auto it = __writers.find("test");
	if(it == __writers.end())
	{
		std::string file_path = __log_dir + "/test/tfevents.pb";
		auto writer = std::make_shared<util::summary_writer>(file_path);
		__writers["test"] = writer;
		__cur_writer = writer;
	}
	else
	{
		__cur_writer = it->second;
	}
}

} // end namespace module
} // end namespace mlfe