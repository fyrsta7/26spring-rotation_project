static inline double read_double_value(Address p) {
  double d;
  memcpy(&d, p, sizeof(d));
  return d;
}
