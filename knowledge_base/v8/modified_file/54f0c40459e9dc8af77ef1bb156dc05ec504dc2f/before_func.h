inline double DoubleToInteger(double x) {
  if (std::isnan(x)) return 0;
  if (!std::isfinite(x)) return x;
  // ToIntegerOrInfinity normalizes -0 to +0, so add 0.0.
  return ((x >= 0) ? std::floor(x) : std::ceil(x)) + 0.0;
}
