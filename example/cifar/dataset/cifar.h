#ifndef __CIFAR10_H__
#define __CIFAR10_H__
#include <vector>
#include <string>
#include <tuple>
#include <mlfe/module/data_gen.h>

namespace dataset{

void read_cifar10_dataset(std::string folder_path,
	std::vector<uint8_t> &train_x,
	std::vector<uint8_t> &train_y,
	std::vector<uint8_t> &valid_x,
	std::vector<uint8_t> &valid_y);

template <int _BatchSize>
struct cifar10_gen : public mlfe::module::generator<cifar10_gen<_BatchSize>>
{
	cifar10_gen(std::vector<uint8_t> x, std::vector<uint8_t> y, bool y_is_x = false)
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

	std::tuple<std::vector<float>, std::vector<float>> call(int batch_idx)
	{
		std::vector<float> x, y;
		constexpr int batch_size = _BatchSize;
		constexpr int classes = 10;
		constexpr int img_size = 3 * 32 * 32;
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

#endif // end #ifndef __CIFAR10_H__