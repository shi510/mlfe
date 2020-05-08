#include "mlfe/module/callbacks/tensorboard.h"
#include "mlfe/module/model.h"
#include "mlfe/utils/string_parser.h"
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
	for(auto& val : logs)
	{
		auto name = string_parser(val.first).split("/").item(-1);
		auto path = string_parser(val.first).remove(name).item(0);
		auto it = __writers.find(path);
		writer_ptr writer;
		if(it == __writers.end())
		{
			std::string file_path = __log_dir+"/"+path+"tfevents.pb";
			writer = std::make_shared<util::summary_writer>(file_path);
			__writers[path] = writer;
		}
		else
		{
			writer = it->second;
		}
		writer->add_scalar(name, epoch, val.second);
	}
}

} // end namespace module
} // end namespace mlfe