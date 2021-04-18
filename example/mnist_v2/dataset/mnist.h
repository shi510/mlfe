#pragma once
#include <vector>
#include <string>
#include <tuple>
#include <mlfe/module/data_gen.h>

namespace dataset
{

using namespace mlfe::module;

void read_mnist_dataset(std::string folder_path,
	std::vector<uint8_t>& train_x,
	std::vector<uint8_t>& train_y,
	std::vector<uint8_t>& valid_x,
	std::vector<uint8_t>& valid_y);

struct mnist_gen : public generator<mnist_gen>
{
	mnist_gen(std::vector<uint8_t> x, std::vector<uint8_t> y, bool y_is_x = false)
	{
		__data_size = y.size();
		__x = x;
		__y = y;
		__y_is_x = y_is_x;
	}

	size_t size() const
	{
		return __data_size;
	}

	std::tuple<std::vector<float>, std::vector<float>> call(int idx)
	{
		std::vector<float> x, y;
		constexpr int classes = 10;
		constexpr int img_size = 1 * 28 * 28;
		const auto iter_beg = __x.data() + int(idx * img_size);
		auto label = (int)__y[idx];
		x.resize(img_size);
		y.resize(classes);
		std::fill(y.begin(), y.end(), 0.f);
		// one-hot.
		y[label] = 1.f;
		// convert to float and divide by 255.
		for(int i = 0; i < img_size; ++i)
		{
			x[i] = (float)iter_beg[i] / 255.f;
		}
		if (__y_is_x)
		{
			return std::make_tuple(x, x);
		}
		return std::make_tuple(x, y);
	}

	size_t __data_size;
	std::vector<uint8_t> __x;
	std::vector<uint8_t> __y;
	bool __y_is_x;
};
	
} // end namespace dataset