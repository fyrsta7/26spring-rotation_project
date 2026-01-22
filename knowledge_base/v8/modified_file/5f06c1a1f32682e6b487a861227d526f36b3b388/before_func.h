bool JSArray::SetLengthWouldNormalize(Heap* heap, uint32_t new_length) {
  // If the new array won't fit in a some non-trivial fraction of the max old
  // space size, then force it to go dictionary mode.
  uint32_t max_fast_array_size =
      static_cast<uint32_t>((heap->MaxOldGenerationSize() / kDoubleSize) / 4);
  return new_length >= max_fast_array_size;
}
