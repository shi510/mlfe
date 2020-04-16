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

template <int _BatchSize>
struct mnist_gen : public generator<mnist_gen<_BatchSize>>
{
	mnist_gen(std::vector<uint8_t> x, std::vector<uint8_t> y)
	{
		__data_size = y.size();
		__x = x;
		__y = y;
	}

	size_t size() const
	{
		return __data_size;
	}

	std::tuple<std::vector<float>, std::vector<float>> call(int batch_idx)
	{
		std::vector<float> x, y;
		constexpr int batch_size = _BatchSize;
		constexpr int classes = 10;
		constexpr int img_size = 1 * 28 * 28;
		const auto iter_beg = __x.data() + batch_idx * batch_size * img_size;
		x.resize(batch_size * img_size);
		y.resize(batch_size * classes);
		std::fill(y.begin(), y.end(), 0.f);
		for(int n = 0; n < batch_size; ++n)
		{
			auto label = (int)__y[batch_idx * batch_size + n];
			// one-hot.
			y[n * classes + label] = 1.f;
			// convert to float and divide by 255.
			for(int i = 0; i < img_size; ++i)
			{
				x[n * img_size + i] = 
					(float)iter_beg[n * img_size + i] / 255.f;
			}
		}
		return std::make_tuple(x, y);
	}

	size_t __data_size;
	std::vector<uint8_t> __x;
	std::vector<uint8_t> __y;
};
	
} // end namespace dataset