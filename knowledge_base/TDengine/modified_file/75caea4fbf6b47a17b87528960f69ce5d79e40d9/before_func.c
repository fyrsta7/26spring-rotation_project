}

static const signed char index_58[256] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,
    -1, -1, -1, -1, -1, -1, -1, 9,  10, 11, 12, 13, 14, 15, 16, -1, 17, 18, 19, 20, 21, -1, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, -1, -1, -1, -1, -1, -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -1, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

uint8_t *base58_decode(const char *value, size_t inlen, int32_t *outlen) {
  const char *pe = value + inlen;
  uint8_t     buf[TBASE_BUF_SIZE] = {0};
  uint8_t    *pbuf = &buf[0];
  bool        bfree = false;
  int32_t     nz = 0, size = 0, len = 0;

  if (inlen > TBASE_MAX_OLEN) {
    terrno = TSDB_CODE_INVALID_PARA;
    return NULL;
  }

  for (int32_t i = 0; i < inlen; ++i) {
    if (value[i] == 0) {
      terrno = TSDB_CODE_INVALID_PARA;
      return NULL;
    }
  }

  while (*value && isspace(*value)) ++value;
  while (*value == '1') {
    ++nz;
    ++value;
  }

  size = (int32_t)(pe - value) * 733 / 1000 + 1;
  if (size > TBASE_BUF_SIZE) {
    if (!(pbuf = taosMemoryCalloc(1, size))) {
      terrno = TSDB_CODE_OUT_OF_MEMORY;
      return NULL;
    }
    bfree = true;
  }

  while (*value && !isspace(*value)) {
    int32_t num = index_58[(uint8_t)*value];
    if (num == -1) {
      terrno = TSDB_CODE_INVALID_PARA;
      if (bfree) taosMemoryFree(pbuf);
      return NULL;
    }
    int32_t i = 0;
    for (int32_t j = size - 1; (num != 0 || i < len) && (j >= 0); --j, ++i) {
      num += (int32_t)pbuf[j] * 58;
      pbuf[j] = num & 255;
      num >>= 8;
    }
    len = i;
    ++value;
  }

  while (isspace(*value)) ++value;
  if (*value != 0) {
    if (bfree) taosMemoryFree(pbuf);
    return NULL;
  }
  const uint8_t *it = pbuf + (size - len);
  while (it != pbuf + size && *it == 0) ++it;

  uint8_t *result = taosMemoryCalloc(1, inlen + 1);
  if (!result) {
