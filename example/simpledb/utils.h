#ifndef __SIMPLEDB_UTILS_H__
#define __SIMPLEDB_UTILS_H__
#include <vector>
#include <string>
#include <mlfe/flatbuffers/tensor_blob_fb_generated.h>

flatbuffers::Offset<mlfe::serializable::TensorBlob>
Serialize(flatbuffers::FlatBufferBuilder &fbb,
          std::string name,
          const uint8_t *data_ptr,
          std::vector<int> dim
         );

#endif // end #ifndef __SIMPLEDB_UTILS_H__