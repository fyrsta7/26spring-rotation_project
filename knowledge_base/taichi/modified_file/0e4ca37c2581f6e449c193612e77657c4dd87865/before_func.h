  return !is_normal(m);
}

inline int64 get_largest_pot(int64 a) noexcept {
  TI_ASSERT_INFO(a > 0,
                 "a should be positive, instead of " + std::to_string(a));
  // TODO: optimize
  int64 i = 1;
  while (i <= a / 2) {
    i *= 2;
