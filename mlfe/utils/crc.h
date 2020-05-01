#pragma once

namespace mlfe{
namespace util{

int crc32file(char *name, uint32_t *crc, long *charcnt);
uint32_t crc32buf(const char *buf, size_t len);
uint32_t masked_crc32c(const char *buf, size_t len);

} // end namespace util
} // end namespace mlfe