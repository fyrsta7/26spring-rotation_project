Stack::StackSlot Stack::ObtainCurrentThreadStackStart() {
#if defined(V8_LIBC_GLIBC)
  // Prefer using __libc_stack_end if it exists and is initialized, since it is
  // generally faster and provides a tighter limit for CSS. Otherwise we
  // fallback to pthread_getattr_np, which can fail for the main thread (See
  // https://code.google.com/p/nativeclient/issues/detail?id=3431).
  if (__libc_stack_end) return __libc_stack_end;
#endif  // !defined(V8_LIBC_GLIBC)

  pthread_attr_t attr;
  int error = pthread_getattr_np(pthread_self(), &attr);
  if (!error) {
    void* base;
    size_t size;
    error = pthread_attr_getstack(&attr, &base, &size);
    CHECK(!error);
    pthread_attr_destroy(&attr);
    return reinterpret_cast<uint8_t*>(base) + size;
  }
  return nullptr;
}
