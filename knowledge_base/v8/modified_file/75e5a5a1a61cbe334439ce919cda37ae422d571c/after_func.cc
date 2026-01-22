bool Sampler::CanSampleOnProfilerEventsProcessorThread() {
#if defined(USE_SIGNALS)
  return true;
#elif V8_OS_WIN || V8_OS_CYGWIN
  return true;
#else
  return false;
#endif
}
