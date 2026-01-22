inline double DoubleToInteger(double x) {
  // ToIntegerOrInfinity normalizes -0 to +0. Special case 0 for performance.
  if (std::isnan(x) || x == 0.0) return 0;
  if (!std::isfinite(x)) return x;
  // Add 0.0 in the truncation case to ensure this doesn't return -0.
  return ((x > 0) ? std::floor(x) : std::ceil(x)) + 0.0;
}
