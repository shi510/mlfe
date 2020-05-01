#pragma once
// this source file is refered from https://github.com/RustingSword/tensorboard_logger.
#include <fstream>
#include <string>
#include <vector>
#include "mlfe/utils/crc.h"
#include "mlfe/utils/tensorboard/proto/event.pb.h"
#include "mlfe/utils/tensorboard/proto/summary.pb.h"

namespace mlfe{
namespace util{

using tensorflow::Summary;
using tensorflow::Event;

class summary_writer final
{
public:
	summary_writer(std::string log_file);

	~summary_writer();

	int add_scalar(const std::string& tag, int step, float value);

	int add_histogram(const std::string& tag, int step, std::vector<float>& value);

private:
	int generate_default_buckets();

	int add_event(int64_t step, Summary* summary);

	int write(Event& event);

	std::ofstream __f;
	std::vector<double>* __bucket_limits;
};

} // end namesapce util
} // end namespace mlfe