#endif

namespace folly {

namespace detail {

uint32_t crc32c_sw(
    const uint8_t* data, size_t nbytes, uint32_t startingChecksum);
#if FOLLY_SSE_PREREQ(4, 2)

uint32_t crc32_sw(
    const uint8_t* data, size_t nbytes, uint32_t startingChecksum);

// Fast SIMD implementation of CRC-32 for x86 with pclmul
uint32_t crc32_hw(
    const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  uint32_t sum = startingChecksum;
  size_t offset = 0;

  // Process unaligned bytes
  if ((uintptr_t)data & 15) {
    size_t limit = std::min(nbytes, -(uintptr_t)data & 15);
