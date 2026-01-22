  return !is_normal(m);
}

inline int64 get_largest_pot(int64 a) noexcept {
  TI_ASSERT_INFO(a > 0,
                 "a should be positive, instead of " + std::to_string(a));

  /* This code was copied from https://stackoverflow.com/a/20207950 and edited
  It uses loop unrolling, which all (modern) compilers will do. */
  for (int64 i = 1; i < 64; i *= 2) {
