static inline double read_double_value(Address p) {
  // Prevent gcc from using load-double (mips ldc1) on (possibly)
  // non-64-bit aligned address.
  // We assume that the address is 32-bit aligned.
  DCHECK(IsAligned(reinterpret_cast<intptr_t>(p), kInt32Size));
  union conversion {
    double d;
    uint32_t u[2];
  } c;
  c.u[0] = Memory::uint32_at(p);
  c.u[1] = Memory::uint32_at(p + 4);
  return c.d;
}
