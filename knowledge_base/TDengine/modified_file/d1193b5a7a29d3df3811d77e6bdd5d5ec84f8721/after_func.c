#if __AVX512F__
    // todo add it
#endif
  return 0;
}

// todo add later
int32_t tsDecompressFloatImplAvx2(const char *const input, const int32_t nelements, char *const output) {
#if __AVX2__
#endif
  return 0;
}

int32_t tsDecompressTimestampAvx2(const char *const input, const int32_t nelements, char *const output,
                                  bool bigEndian) {
  int64_t *ostream = (int64_t *)output;
  int32_t  ipos = 1, opos = 0;
  __m128i  prevVal = _mm_setzero_si128();
  __m128i  prevDelta = _mm_setzero_si128();

#if __AVX2__
  int32_t batch = nelements >> 1;
  int32_t remainder = nelements & 0x01;
  __mmask16 mask2[16] = {0, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff};

  int32_t i = 0;
  if (batch > 1) {
    // first loop
    uint8_t flags = input[ipos++];

    int8_t nbytes1 = flags & INT8MASK(4);  // range of nbytes starts from 0 to 7
    int8_t nbytes2 = (flags >> 4) & INT8MASK(4);

    __m128i data1;
    if (nbytes1 == 0) {
      data1 = _mm_setzero_si128();
    } else {
//      _mm_shuffle_epi8()
      memcpy(&data1, (const void*) (input + ipos), nbytes1);
    }

    __m128i data2;
    if (nbytes2 == 0) {
      data2 = _mm_setzero_si128();
    } else {
      memcpy(&data2, (const void*) (input + ipos + nbytes1), nbytes2);
    }

    data2 = _mm_broadcastq_epi64(data2);
    __m128i zzVal = _mm_blend_epi32(data2, data1, 0x03);

    // ZIGZAG_DECODE(T, v) (((v) >> 1) ^ -((T)((v)&1)))
    __m128i signmask = _mm_and_si128(_mm_set1_epi64x(1), zzVal);
    signmask = _mm_sub_epi64(_mm_setzero_si128(), signmask);

    // get two zigzag values here
    __m128i deltaOfDelta = _mm_xor_si128(_mm_srli_epi64(zzVal, 1), signmask);

    __m128i deltaCurrent = _mm_add_epi64(deltaOfDelta, prevDelta);
    deltaCurrent = _mm_add_epi64(_mm_slli_si128(deltaCurrent, 8), deltaCurrent);

    __m128i val = _mm_add_epi64(deltaCurrent, prevVal);
    _mm_storeu_si128((__m128i *)&ostream[opos], val);

    // keep the previous value
    prevVal = _mm_shuffle_epi32 (val, 0xEE);

    // keep the previous delta of delta, for the first item
    prevDelta = _mm_shuffle_epi32(deltaOfDelta, 0xEE);

    opos += 2;
    ipos += nbytes1 + nbytes2;
    i += 1;
  }

  // the remain
  for(; i < batch; ++i) {
    uint8_t flags = input[ipos++];

    int8_t nbytes1 = flags & INT8MASK(4);  // range of nbytes starts from 0 to 7
    int8_t nbytes2 = (flags >> 4) & INT8MASK(4);

//    __m128i data1 = _mm_maskz_loadu_epi8(mask2[nbytes1], (const void*)(input + ipos));
//    __m128i data2 = _mm_maskz_loadu_epi8(mask2[nbytes2], (const void*)(input + ipos + nbytes1));
    __m128i data1;
    if (nbytes1 == 0) {
      data1 = _mm_setzero_si128();
    } else {
      int64_t dd = 0;
      memcpy(&dd, (const void*) (input + ipos), nbytes1);
      data1 = _mm_loadu_si64(&dd);
    }

    __m128i data2;
    if (nbytes2 == 0) {
      data2 = _mm_setzero_si128();
    } else {
      int64_t dd = 0;
      memcpy(&dd, (const void*) (input + ipos + nbytes1), nbytes2);
      data2 = _mm_loadu_si64(&dd);
    }

    data2 = _mm_broadcastq_epi64(data2);

    __m128i zzVal = _mm_blend_epi32(data2, data1, 0x03);

    // ZIGZAG_DECODE(T, v) (((v) >> 1) ^ -((T)((v)&1)))
    __m128i signmask = _mm_and_si128(_mm_set1_epi64x(1), zzVal);
    signmask = _mm_sub_epi64(_mm_setzero_si128(), signmask);

    // get two zigzag values here
    __m128i deltaOfDelta = _mm_xor_si128(_mm_srli_epi64(zzVal, 1), signmask);

    __m128i deltaCurrent = _mm_add_epi64(deltaOfDelta, prevDelta);
    deltaCurrent = _mm_add_epi64(_mm_slli_si128(deltaCurrent, 8), deltaCurrent);

    __m128i val = _mm_add_epi64(deltaCurrent, prevVal);
    _mm_storeu_si128((__m128i *)&ostream[opos], val);

    // keep the previous value
    prevVal = _mm_shuffle_epi32 (val, 0xEE);

    // keep the previous delta of delta
    __m128i delta = _mm_add_epi64(_mm_slli_si128(deltaOfDelta, 8), deltaOfDelta);
    prevDelta = _mm_shuffle_epi32(_mm_add_epi64(delta, prevDelta), 0xEE);

    opos += 2;
    ipos += nbytes1 + nbytes2;
  }

  if (remainder > 0) {
    uint64_t dd = 0;
    uint8_t  flags = input[ipos++];

    int32_t nbytes = flags & INT8MASK(4);
    int64_t deltaOfDelta = 0;
    if (nbytes == 0) {
      deltaOfDelta = 0;
    } else {
      //      if (is_bigendian()) {
      //        memcpy(((char *)(&dd1)) + longBytes - nbytes, input + ipos, nbytes);
      //      } else {
      memcpy(&dd, input + ipos, nbytes);
      //      }
      deltaOfDelta = ZIGZAG_DECODE(int64_t, dd);
    }
