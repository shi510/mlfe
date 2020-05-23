#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/blas.h"
#include <cudnn.h>

namespace mlfe{
namespace algorithm_cudnn{

template <class Tp>
class BatchNormSpatial : public OpAlgo
{
	using T = typename Tp::T;

public:
	BatchNormSpatial(OpAlgoContext *oac) : OpAlgo(oac, "BatchNormSpatial")
	{
		auto cudnn_type = [](std::string type_str) {
			return type_str == type::float32::string ? 
				CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
		};
		std::vector<int> x_shape;
		y = oac->get_output(0);
		x = oac->get_input(0);
		x_shape.resize(x.dims());
		std::copy(x.shape().begin(), x.shape().end(), x_shape.begin());
		if(x_shape.size() == 2){
			x_shape.push_back(1);
			x_shape.push_back(1);
		}
		_scales_ptr = create_memory(x_shape[1] * sizeof(T));
		_biases_ptr = create_memory(x_shape[1] * sizeof(T));
		_running_mean_ptr = create_memory(x_shape[1] * sizeof(T));
		_running_variance_ptr = create_memory(x_shape[1] * sizeof(T));
		_mean_ptr = create_memory(x_shape[1] * sizeof(T));
		_variance_ptr = create_memory(x_shape[1] * sizeof(T));
		math::set<T, CUDAContext>(x_shape[1], 1.f, _scales_ptr->mutable_device_data<float>());
		math::set<T, CUDAContext>(x_shape[1], 0.f, _biases_ptr->mutable_device_data<float>());
		oac->add_attr({ "scales", _scales_ptr });
		oac->add_attr({ "biases", _biases_ptr });
		oac->add_attr({ "running_mean", _running_mean_ptr });
		oac->add_attr({ "running_variance", _running_variance_ptr });
		oac->add_attr({ "mean", _mean_ptr });
		oac->add_attr({ "variance", _variance_ptr });
		cudnnCreate(&_handle);
		cudnnCreateTensorDescriptor(&_dst_desc);
		cudnnCreateTensorDescriptor(&_norm_desc);
		cudnnSetTensor4dDescriptor(_dst_desc,
			CUDNN_TENSOR_NCHW,
			cudnn_type(Tp::string),
			x_shape[0], x_shape[1],
			x_shape[2], x_shape[3]);
		cudnnSetTensor4dDescriptor(_norm_desc,
			CUDNN_TENSOR_NCHW,
			cudnn_type(Tp::string),
			1, x_shape[1], 1, 1);
	}

	void Compute(op_algo_runtime_context& rc) override
	{
		const float one = 1;
		const float zero = 0;
		if(rc.training())
		{
			cudnnBatchNormalizationForwardTraining(_handle,
				CUDNN_BATCHNORM_SPATIAL,
				&one,
				&zero,
				_dst_desc,
				x.device_data<void>(),
				_dst_desc,
				y.mutable_device_data<void>(),
				_norm_desc,
				_scales_ptr->device_data<void>(),
				_biases_ptr->device_data<void>(),
				.01,
				_running_mean_ptr->mutable_device_data<void>(),
				_running_variance_ptr->mutable_device_data<void>(),
				.00001,
				_mean_ptr->mutable_device_data<void>(),
				_variance_ptr->mutable_device_data<void>());
		}
		else
		{
			cudnnBatchNormalizationForwardInference(_handle,
				CUDNN_BATCHNORM_SPATIAL,
				&one,
				&zero,
				_dst_desc,
				x.device_data<void>(),
				_dst_desc,
				y.mutable_device_data<void>(),
				_norm_desc,
				_scales_ptr->device_data<void>(),
				_biases_ptr->device_data<void>(),
				_running_mean_ptr->device_data<void>(),
				_running_variance_ptr->device_data<void>(),
				.00001);
		}

	}

	~BatchNormSpatial()
	{
		cudnnDestroyTensorDescriptor(_dst_desc);
		cudnnDestroyTensorDescriptor(_norm_desc);
		cudnnDestroy(_handle);
	}

private:
	Tensor x;
	Tensor y;
	memory_ptr _scales_ptr;
	memory_ptr _biases_ptr;
	memory_ptr _running_mean_ptr;
	memory_ptr _running_variance_ptr;
	memory_ptr _mean_ptr;
	memory_ptr _variance_ptr;
	cudnnHandle_t _handle;
	cudnnTensorDescriptor_t _dst_desc;
	cudnnTensorDescriptor_t _norm_desc;
};

REGIST_OP_ALGO(BatchNormSpatial)
	.Input("X", type::float32::string)
	.Output("Y", type::float32::string)
	.Device("CUDA(CUDNN)")
	.CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo> {
		using T = BatchNormSpatial<type::float32>;
		return std::make_shared<T>(oac);
	})
	.Finish();

template <class Tp>
class BatchNormSpatialGrad : public OpAlgo
{
	using T = typename Tp::T;

public:
	BatchNormSpatialGrad(OpAlgoContext *oac) : OpAlgo(oac, "BatchNormSpatialGrad")
	{
		auto cudnn_type = [](std::string type_str) {
			return type_str == type::float32::string ?
				CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
		};
		std::vector<int> x_shape;
		dx = oac->get_output(0);
		dy = oac->get_input(0);
		x = oac->get_input(1);
		x_shape.resize(x.dims());
		std::copy(x.shape().begin(), x.shape().end(), x_shape.begin());
		if(x_shape.size() == 2){
			x_shape.push_back(1);
			x_shape.push_back(1);
		}
		_scales_ptr = oac->get_attr<memory_ptr>("scales");
		_biases_ptr = oac->get_attr<memory_ptr>("biases");
		_running_mean_ptr = oac->get_attr<memory_ptr>("running_mean");
		_running_variance_ptr = oac->get_attr<memory_ptr>("running_variance");
		_mean_ptr = oac->get_attr<memory_ptr>("mean");
		_variance_ptr = oac->get_attr<memory_ptr>("variance");
		_scales_diff_ptr = create_memory(x_shape[1] * sizeof(T));
		_biases_diff_ptr = create_memory(x_shape[1] * sizeof(T));
		cudnnCreate(&_handle);
		cudnnCreateTensorDescriptor(&_dst_desc);
		cudnnCreateTensorDescriptor(&_norm_desc);
		cudnnSetTensor4dDescriptor(_dst_desc,
			CUDNN_TENSOR_NCHW,
			cudnn_type(Tp::string),
			x_shape[0], x_shape[1],
			x_shape[2], x_shape[3]);
		cudnnSetTensor4dDescriptor(_norm_desc,
			CUDNN_TENSOR_NCHW,
			cudnn_type(Tp::string),
			1, x_shape[1], 1, 1);
	}

	void Compute(op_algo_runtime_context& rc) override
	{
		const float one = 1;
		const float zero = 0;
		cudnnBatchNormalizationBackward(_handle,
			CUDNN_BATCHNORM_SPATIAL,
			&one,
			&zero,
			&one,
			&one,
			_dst_desc,
			x.device_data<void>(),
			_dst_desc,
			dy.device_data<void>(),
			_dst_desc,
			dx.mutable_device_data<void>(),
			_norm_desc,
			_scales_ptr->device_data<void>(),
			_scales_diff_ptr->mutable_device_data<void>(),
			_biases_diff_ptr->mutable_device_data<void>(),
			.00001,
			_mean_ptr->mutable_device_data<void>(),
			_variance_ptr->mutable_device_data<void>());
	}

	~BatchNormSpatialGrad()
	{
		cudnnDestroyTensorDescriptor(_dst_desc);
		cudnnDestroyTensorDescriptor(_norm_desc);
		cudnnDestroy(_handle);
	}

private:
	Tensor dx;
	Tensor dy;
	Tensor x;
	memory_ptr _scales_diff_ptr;
	memory_ptr _biases_diff_ptr;
	memory_ptr _scales_ptr;
	memory_ptr _biases_ptr;
	memory_ptr _running_mean_ptr;
	memory_ptr _running_variance_ptr;
	memory_ptr _mean_ptr;
	memory_ptr _variance_ptr;
	cudnnHandle_t _handle;
	cudnnTensorDescriptor_t _dst_desc;
	cudnnTensorDescriptor_t _norm_desc;
};

REGIST_OP_GRAD_ALGO(BatchNormSpatial)
	.Input("X", type::float32::string)
	.Input("dY", type::float32::string)
	.Output("dX", type::float32::string)
	.Device("CUDA(CUDNN)")
	.CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo> {
		using T = BatchNormSpatialGrad<type::float32>;
		return std::make_shared<T>(oac);
	})
	.Finish();

} // end namespace algorithm_cudnn
} // end namespace mlfe
