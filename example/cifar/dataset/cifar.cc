#include "cifar.h"
#include <fstream>
#include <iostream>

namespace dataset
{

void read_all(std::vector<std::string> train_batch_names,
	std::vector<unsigned char> &data,
	std::vector<unsigned char> &label)
{
	const int img_c = 3;
	const int img_h = 32;
	const int img_w = 32;
	const int img_size = img_c * img_h * img_w;
	const int num_batch = train_batch_names.size();
	const int num_data = 10000;
	std::ifstream train_file;

	data.resize(num_batch * num_data * img_size);
	label.resize(num_batch * num_data);
	if(data.size() != num_batch * num_data * img_size)
	{
		throw std::string("Can not allocate memory for data size : ") +
			std::to_string(num_batch * num_data * img_size);
	}
	if(label.size() != num_batch * num_data)
	{
		throw std::string("Can not allocate memory for label size : ") +
			std::to_string(num_batch * num_data);
	}
	for(int n = 0; n < num_batch; ++n)
	{
		int cur_batch = n * num_data;
		train_file.open(train_batch_names[n], std::ios::binary);
		if(!train_file.is_open())
		{
			throw std::string("can not open file : ") + train_batch_names[n];
		}
		
		for(int i = 0; i < num_data; ++i)
		{
			auto data_ptr = data.data() + cur_batch * img_size + i * img_size;
			auto label_ptr = label.data() + cur_batch + i;
			train_file.read((char *)label_ptr, 1);
			train_file.read((char *)data_ptr, img_size);
		}
		train_file.close();
	}
}

void read_cifar10_dataset(std::string folder_path,
	std::vector<uint8_t> &train_x,
	std::vector<uint8_t> &train_y,
	std::vector<uint8_t> &valid_x,
	std::vector<uint8_t> &valid_y)
{
	std::vector<std::string> train_files =
	{
		folder_path + "/data_batch_1.bin",
		folder_path + "/data_batch_2.bin",
		folder_path + "/data_batch_3.bin",
		folder_path + "/data_batch_4.bin",
		folder_path + "/data_batch_5.bin"
	};

	read_all(train_files, train_x, train_y);
	read_all({folder_path + "/test_batch.bin"}, valid_x, valid_y);
}

} // end namespace dataset