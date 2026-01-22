void SharedTurboAssembler::F64x2ReplaceLane(XMMRegister dst, XMMRegister src,
                                            DoubleRegister rep, uint8_t lane) {
  if (CpuFeatures::IsSupported(AVX)) {
    CpuFeatureScope scope(this, AVX);
    if (lane == 0) {
      vpblendw(dst, src, rep, 0b00001111);
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
      pblendw(dst, rep, 0b00001111);
    } else {
      movlhps(dst, rep);
    }
  }
}
