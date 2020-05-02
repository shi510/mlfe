#pragma once
#include "mlfe/module/callbacks/callback.h"
#include "mlfe/utils/tensorboard/summary_writer.h"
#include <string>

namespace mlfe{
namespace module{

class tensorboard : public callback
{
public:
	tensorboard(std::string log_dir);

	void on_epoch_end(const int epoch,
		const std::map<std::string, float>& logs) override;

	void on_train_begin(const int epoch,
		const std::map<std::string, float>& logs) override;

	void on_test_begin(const int epoch,
		const std::map<std::string, float>& logs) override;

private:
	using writer_ptr = std::shared_ptr<util::summary_writer>;
	using writer_map = std::map<std::string, writer_ptr>;
	std::string __log_dir;
	writer_map __writers;
	writer_ptr __cur_writer;
};

} // end namespace module
} // end namespace mlfe