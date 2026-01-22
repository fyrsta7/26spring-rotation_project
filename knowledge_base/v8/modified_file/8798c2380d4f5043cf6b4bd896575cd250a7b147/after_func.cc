void SharedTurboAssembler::F64x2ReplaceLane(XMMRegister dst, XMMRegister src,
                                            DoubleRegister rep, uint8_t lane) {
  if (CpuFeatures::IsSupported(AVX)) {
    CpuFeatureScope scope(this, AVX);
    if (lane == 0) {
      vmovsd(dst, src, rep);
    } else {
      vmovlhps(dst, src, rep);
    }
  } else {
    CpuFeatureScope scope(this, SSE4_1);
    if (dst != src) {
      DCHECK_NE(dst, rep);  // Ensure rep is not overwritten.
      movaps(dst, src);
    }
    if (lane == 0) {
      movsd(dst, rep);
    } else {
      movlhps(dst, rep);
    }
  }
}
