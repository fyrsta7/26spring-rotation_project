SIMDDetect::SIMDDetect() {
  // The fallback is a generic dot product calculation.
  SetDotProduct(DotProductGeneric);

#if defined(HAS_CPUID)
#if defined(__GNUC__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0) {
    // Note that these tests all use hex because the older compilers don't have
    // the newer flags.
#if defined(SSE4_1)
    sse_available_ = (ecx & 0x00080000) != 0;
#endif
#if defined(FMA)
    fma_available_ = (ecx & 0x00001000) != 0;
#endif
#if defined(AVX)
    avx_available_ = (ecx & 0x10000000) != 0;
    if (avx_available_) {
      // There is supposed to be a __get_cpuid_count function, but this is all
      // there is in my cpuid.h. It is a macro for an asm statement and cannot
      // be used inside an if.
      __cpuid_count(7, 0, eax, ebx, ecx, edx);
      avx2_available_ = (ebx & 0x00000020) != 0;
      avx512F_available_ = (ebx & 0x00010000) != 0;
      avx512BW_available_ = (ebx & 0x40000000) != 0;
    }
#endif
  }
#  elif defined(_WIN32)
  int cpuInfo[4];
  int max_function_id;
  __cpuid(cpuInfo, 0);
  max_function_id = cpuInfo[0];
  if (max_function_id >= 1) {
    bool xmm_ymm_state_enabled = (_xgetbv(0) & 6) == 6;
    __cpuid(cpuInfo, 1);
#if defined(SSE4_1)
    sse_available_ = (cpuInfo[2] & 0x00080000) != 0;
#endif
#if defined(FMA)
    fma_available_ = xmm_ymm_state_enabled && (cpuInfo[2] & 0x00001000) != 0;
#endif
#if defined(AVX)
    avx_available_ = (cpuInfo[2] & 0x10000000) != 0;
#endif
#if defined(AVX2)
    if (max_function_id >= 7 && xmm_ymm_state_enabled) {
      __cpuid(cpuInfo, 7);
      avx2_available_ = (cpuInfo[1] & 0x00000020) != 0;
      avx512F_available_ = (cpuInfo[1] & 0x00010000) != 0;
      avx512BW_available_ = (cpuInfo[1] & 0x40000000) != 0;
    }
#endif
  }
#else
#error "I don't know how to test for SIMD with this compiler"
#endif
#endif

  // Select code for calculation of dot product based on autodetection.
  if (false) {
    // This is a dummy to support conditional compilation.
#if defined(AVX2)
  } else if (avx2_available_) {
    // AVX2 detected.
    SetDotProduct(DotProductAVX, &IntSimdMatrix::intSimdMatrixAVX2);
#endif
#if defined(AVX)
  } else if (avx_available_) {
    // AVX detected.
    SetDotProduct(DotProductAVX, &IntSimdMatrix::intSimdMatrixSSE);
#endif
#if defined(SSE4_1)
  } else if (sse_available_) {
    // SSE detected.
    SetDotProduct(DotProductSSE, &IntSimdMatrix::intSimdMatrixSSE);
#endif
  }
}
