float DotProductFMA(const float *u, const float *v, int n) {
  const unsigned quot = n / 8;
  const unsigned rem = n % 8;
  __m256 t0 = _mm256_setzero_ps();
  __m256 t1 = _mm256_setzero_ps();
  for (unsigned k = 0; k < quot; k++) {
    __m256 f0 = _mm256_loadu_ps(u);
    __m256 f1 = _mm256_loadu_ps(v);
    t0 = _mm256_fmadd_ps(f0, f1, t0);
    u += 4;
    v += 4;
    __m256 f2 = _mm256_loadu_ps(u);
    __m256 f3 = _mm256_loadu_ps(v);
    t1 = _mm256_fmadd_ps(f2, f3, t1);
    u += 4;
    v += 4;
  }
  t0 = _mm256_hadd_ps(t0, t1);
  alignas(32) float tmp[4];
  _mm256_store_ps(tmp, t0);
  float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  for (unsigned k = 0; k < rem; k++) {
    result += *u++ * *v++;
  }
  return result;
}
