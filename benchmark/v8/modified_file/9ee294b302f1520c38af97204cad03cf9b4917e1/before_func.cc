// Returns a number value with positive sign, greater than or equal to
// 0 but less than 1, chosen randomly.
static Object* Runtime_Math_random(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 0);

  // To get much better precision, we combine the results of two
  // invocations of random(). The result is computed by normalizing a
  // double in the range [0, RAND_MAX + 1) obtained by adding the
  // high-order bits in the range [0, RAND_MAX] with the low-order
  // bits in the range [0, 1).
  double lo = static_cast<double>(random()) / (RAND_MAX + 1.0);
  double hi = static_cast<double>(random());
  double result = (hi + lo) / (RAND_MAX + 1.0);
  ASSERT(result >= 0 && result < 1);
  return Heap::AllocateHeapNumber(result);
}