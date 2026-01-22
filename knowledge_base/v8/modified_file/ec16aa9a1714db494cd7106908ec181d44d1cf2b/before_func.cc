static void ClearSampleBufferNewSpaceEntries() {
  for (int i = 0; i < kSamplerWindowSize; i++) {
    if (Heap::InNewSpace(sampler_window[i])) {
      sampler_window[i] = NULL;
      sampler_window_weight[i] = 0;
    }
  }
}
