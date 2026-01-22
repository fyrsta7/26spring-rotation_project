// Result is four int32 scalars packed into a XMM register.
// int8x4x4 · int8x4x4 => int32x4
static inline __m128i DotProdInt8x4x4(__m128i a_8x16, __m128i b_8x16) {
  // Transfer sign from 'a' to 'b', as _mm_maddubs_epi16 treats 'a' unsigned.
  b_8x16 = _mm_sign_epi8(b_8x16, a_8x16);
  a_8x16 = _mm_abs_epi8(a_8x16);
  // sumprod[i] = a[2*i]*b[2*i] + a[2*i+1]*b[2*i+1] (i = 0..7)
  __m128i sumprod_16x8 = _mm_maddubs_epi16(a_8x16, b_8x16);
  // sumprod[i] = sumprod[2*i]*1 + sumprod[2*i+1]*1 (i = 0..3)
  return _mm_madd_epi16(sumprod_16x8, _mm_set1_epi16(1));
}

