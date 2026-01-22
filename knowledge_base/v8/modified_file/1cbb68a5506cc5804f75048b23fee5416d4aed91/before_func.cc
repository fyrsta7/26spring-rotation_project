void SharedTurboAssembler::I64x2UConvertI32x4High(XMMRegister dst,
                                                  XMMRegister src,
                                                  XMMRegister scratch) {
  if (CpuFeatures::IsSupported(AVX)) {
    CpuFeatureScope avx_scope(this, AVX);
    vpxor(scratch, scratch, scratch);
    vpunpckhdq(dst, src, scratch);
  } else {
    if (dst != src) {
      movaps(dst, src);
    }
    xorps(scratch, scratch);
    punpckhdq(dst, scratch);
  }
}
