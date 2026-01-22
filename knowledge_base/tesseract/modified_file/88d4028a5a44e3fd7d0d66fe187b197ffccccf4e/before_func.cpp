double DotProductNative(const double *u, const double *v, int n) {
  double total = 0.0;
#if defined(OPENMP_SIMD)
#pragma omp simd reduction(+:total)
#endif
  for (int k = 0; k < n; ++k) {
    total += u[k] * v[k];
  }
  return total;
}
