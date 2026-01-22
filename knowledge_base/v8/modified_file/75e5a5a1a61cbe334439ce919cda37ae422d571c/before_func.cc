bool Sampler::CanSampleOnProfilerEventsProcessorThread() {
#if defined(USE_SIGNALS)
  return true;
#else
  return false;
#endif
}
