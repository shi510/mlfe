#include "mlfe/utils/tensorboard/summary_writer.h"
#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

namespace mlfe{
namespace util{

using tensorflow::Summary;
using tensorflow::Event;
using tensorflow::HistogramProto;

summary_writer::summary_writer(std::string log_file)
{
	__file_path = log_file;
	__bucket_limits = NULL;
	if(!std::filesystem::exists(log_file))
	{
		std::filesystem::create_directories(
			std::filesystem::path(log_file).parent_path());
	}
}

summary_writer::~summary_writer()
{
	if(__f.is_open())
	{
		__f.close();
	}
	if(__bucket_limits != NULL)
	{
		delete __bucket_limits;
		__bucket_limits = NULL;
	}
}

int summary_writer::add_histogram(const std::string& tag,
	const int step, const std::vector<float>& values)
{
	if(__bucket_limits == NULL)
	{
		generate_default_buckets();
	}

	std::vector<int> counts(__bucket_limits->size(), 0);
	double min = std::numeric_limits<double>::max();
	double max = std::numeric_limits<double>::lowest();
	double sum = 0.0;
	double sum_squares = 0.0;
	for(auto v: values)
	{
		auto lb = lower_bound(__bucket_limits->begin(), __bucket_limits->end(), v);
		counts[lb - __bucket_limits->begin()]++;
		sum += v;
		sum_squares += v * v;
		if(v > max)
		{
			max = v;
		}
		else if(v < min)
		{
			min = v;
		}
	}

	auto histo = new HistogramProto();
	histo->set_min(min);
	histo->set_max(max);
	histo->set_num(values.size());
	histo->set_sum(sum);
	histo->set_sum_squares(sum_squares);
	for(size_t i = 0; i < counts.size(); ++i)
	{
		if(counts[i] > 0)
		{
			histo->add_bucket_limit((*__bucket_limits)[i]);
			histo->add_bucket(counts[i]);
		}
	}

	auto summary = new Summary();
	auto v = summary->add_value();
	v->set_node_name(tag);
	v->set_tag(tag);
	v->set_allocated_histo(histo);

	return add_event(step, summary);
}

int summary_writer::add_scalar(const std::string& tag, const int step,
	const float value)
{
	auto summary = new Summary();
	auto v = summary->add_value();
	v->set_node_name(tag);
	v->set_tag(tag);
	v->set_simple_value(value);
	return add_event(step, summary);
}

int summary_writer::generate_default_buckets()
{
	if(__bucket_limits == NULL)
	{
		__bucket_limits = new std::vector<double>;
		std::vector<double> pos_buckets, neg_buckets;
		double v = 1e-12;
		while (v < 1e20)
		{
			pos_buckets.push_back(v);
			neg_buckets.push_back(-v);
			v *= 1.1;
		}
		pos_buckets.push_back(std::numeric_limits<double>::max());
		neg_buckets.push_back(std::numeric_limits<double>::lowest());

		__bucket_limits->insert(__bucket_limits->end(),
			neg_buckets.rbegin(),
			neg_buckets.rend());
		__bucket_limits->insert(__bucket_limits->end(),
			pos_buckets.begin(),
			pos_buckets.end());
	}

	return 0;
}

int summary_writer::write(Event& event)
{
	std::string buf;
	event.SerializeToString(&buf);
	uint64_t buf_len = static_cast<uint64_t>(buf.size());
	uint32_t len_crc = masked_crc32c((char*)&buf_len, sizeof(uint64_t));  // NOLINT
	uint32_t data_crc = masked_crc32c(buf.c_str(), buf.size());

	__f.open(__file_path,
		std::ios::out | std::ios::app | std::ios::binary);
	if(!__f.is_open())
	{
		std::cout<<"Fail to open "<<__file_path<<std::endl;
	}
	else
	{
		__f.write((char*)&buf_len, sizeof(uint64_t));  // NOLINT
		__f.write((char*)&len_crc, sizeof(uint32_t));  // NOLINT
		__f.write(buf.c_str(), buf.size());
		__f.write((char*)&data_crc, sizeof(uint32_t));  // NOLINT
		__f.flush();
		__f.close();
	}

	return 0;
}

int summary_writer::add_event(int64_t step, Summary* summary)
{
	using namespace std::chrono;
	Event event;
	double wall_time = time(nullptr);
	event.set_wall_time(wall_time);
	event.set_step(step);
	event.set_allocated_summary(summary);
	return write(event);
}

} // end namespace util
} // end namespace mlfe