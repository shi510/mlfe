#include "utils.h"
#include <numeric>

typedef mlfe::serializable::TensorBlob SerializableTensor;
typedef flatbuffers::FlatBufferBuilder FBB;
typedef flatbuffers::Offset<SerializableTensor> OffsetSerialTensor;

OffsetSerialTensor Serialize(FBB &fbb,
                             std::string name,
                             const uint8_t *data_ptr,
                             std::vector<int> dim
                            )
{
    int size;
    size = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<int>());
    auto name_fb = fbb.CreateString(name);
    auto data_fb = fbb.CreateVector(data_ptr, size * sizeof(uint8_t));
    auto dim_fb = fbb.CreateVector(dim.data(), dim.size());
    auto tb_fb = mlfe::serializable::CreateTensorBlob(fbb, name_fb, data_fb, dim_fb);
    return tb_fb;
}
