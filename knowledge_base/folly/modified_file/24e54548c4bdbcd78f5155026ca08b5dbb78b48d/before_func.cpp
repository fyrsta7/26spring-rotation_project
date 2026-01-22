void toLowerAscii64(uint64_t& c) {
  // 64-bit version of toLower32
  uint64_t rotated = c & uint64_t(0x7f7f7f7f7f7f7f7fL);
  rotated += uint64_t(0x2525252525252525L);
  rotated &= uint64_t(0x7f7f7f7f7f7f7f7fL);
  rotated += uint64_t(0x1a1a1a1a1a1a1a1aL);
  rotated &= ~c;
  rotated >>= 2;
  rotated &= uint64_t(0x2020202020202020L);
  c += rotated;
}

} // anon namespace

void toLowerAscii(char* str, size_t length) {
  static const size_t kAlignMask64 = 7;
  static const size_t kAlignMask32 = 3;

  // Convert a character at a time until we reach an address that
  // is at least 32-bit aligned
  size_t n = (size_t)str;
  n &= kAlignMask32;
  n = std::min(n, length);
  size_t offset = 0;
  if (n != 0) {
    do {
      toLowerAscii8(str[offset]);
      offset++;
    } while (offset < n);
  }

  n = (size_t)(str + offset);
  n &= kAlignMask64;
  if ((n != 0) && (offset + 4 < length)) {
    // The next address is 32-bit aligned but not 64-bit aligned.
    // Convert the next 4 bytes in order to get to the 64-bit aligned
    // part of the input.
    toLowerAscii32(*(uint32_t*)(str + offset));
    offset += 4;
  }

  // Convert 8 characters at a time
  while (offset + 8 < length) {
    toLowerAscii64(*(uint64_t*)(str + offset));
    offset += 8;
