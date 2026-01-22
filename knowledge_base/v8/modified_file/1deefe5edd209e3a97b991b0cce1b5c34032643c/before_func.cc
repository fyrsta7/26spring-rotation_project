RandomNumberGenerator::RandomNumberGenerator() {
  // Check if embedder supplied an entropy source.
  {
    MutexGuard lock_guard(entropy_mutex.Pointer());
    if (entropy_source != nullptr) {
      int64_t seed;
      if (entropy_source(reinterpret_cast<unsigned char*>(&seed),
                         sizeof(seed))) {
        SetSeed(seed);
        return;
      }
    }
  }

#if V8_OS_CYGWIN || V8_OS_WIN
  // Use rand_s() to gather entropy on Windows. See:
  // https://code.google.com/p/v8/issues/detail?id=2905
  unsigned first_half, second_half;
  errno_t result = rand_s(&first_half);
  DCHECK_EQ(0, result);
  result = rand_s(&second_half);
  DCHECK_EQ(0, result);
  SetSeed((static_cast<int64_t>(first_half) << 32) + second_half);
#elif V8_OS_MACOSX
  // Despite its prefix suggests it is not RC4 algorithm anymore.
  // It always succeeds while having decent performance and
  // no file descriptor involved.
  int64_t seed;
  arc4random_buf(&seed, sizeof(seed));
  SetSeed(seed);
#else
  // Gather entropy from /dev/urandom if available.
  FILE* fp = fopen("/dev/urandom", "rb");
  if (fp != nullptr) {
    int64_t seed;
    size_t n = fread(&seed, sizeof(seed), 1, fp);
    fclose(fp);
    if (n == 1) {
      SetSeed(seed);
      return;
    }
  }

  // We cannot assume that random() or rand() were seeded
  // properly, so instead of relying on random() or rand(),
  // we just seed our PRNG using timing data as fallback.
  // This is weak entropy, but it's sufficient, because
  // it is the responsibility of the embedder to install
  // an entropy source using v8::V8::SetEntropySource(),
  // which provides reasonable entropy, see:
  // https://code.google.com/p/v8/issues/detail?id=2905
  int64_t seed = Time::NowFromSystemTime().ToInternalValue() << 24;
  seed ^= TimeTicks::HighResolutionNow().ToInternalValue() << 16;
  seed ^= TimeTicks::Now().ToInternalValue() << 8;
  SetSeed(seed);
#endif  // V8_OS_CYGWIN || V8_OS_WIN
}
