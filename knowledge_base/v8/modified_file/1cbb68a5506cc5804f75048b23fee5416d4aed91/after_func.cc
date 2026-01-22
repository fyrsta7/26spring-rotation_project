void SharedTurboAssembler::I64x2UConvertI32x4High(XMMRegister dst,
                                                  XMMRegister src,
                                                  XMMRegister scratch) {
  if (CpuFeatures::IsSupported(AVX)) {
    CpuFeatureScope avx_scope(this, AVX);
    vpxor(scratch, scratch, scratch);
    vpunpckhdq(dst, src, scratch);
  } else {
    if (dst == src) {
      // xorps can be executed on more ports than pshufd.
      xorps(scratch, scratch);
      punpckhdq(dst, scratch);
    } else {
      CpuFeatureScope sse_scope(this, SSE4_1);
      // No dependency on dst.
      pshufd(dst, src, 0xEE);
      pmovzxdq(dst, dst);
    }
  }
}
