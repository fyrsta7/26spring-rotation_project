bool JSArray::SetLengthWouldNormalize(Heap* heap, uint32_t new_length) {
  // This constant is somewhat arbitrary. Any large enough value would work.
  const uint32_t kMaxFastArrayLength = 32 * 1024 * 1024;
  // If the new array won't fit in a some non-trivial fraction of the max old
  // space size, then force it to go dictionary mode.
  uint32_t heap_based_upper_bound =
      static_cast<uint32_t>((heap->MaxOldGenerationSize() / kDoubleSize) / 4);
  return new_length >= Min(kMaxFastArrayLength, heap_based_upper_bound);
}
