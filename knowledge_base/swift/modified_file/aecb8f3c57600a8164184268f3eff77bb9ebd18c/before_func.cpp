__attribute__((constructor))
static void installGetClassHook_untrusted() {
  // swiftCompatibility* might be linked into multiple dynamic libraries because
  // of build system reasons, but the copy in the main executable is the only
  // one that should count. Bail early unless we're running out of the main
  // executable.
  //
  // Newer versions of dyld add additional API that can determine this more
  // efficiently, but we have to support back to OS X 10.9/iOS 7, so dladdr
  // is the only API that reaches back that far.
  Dl_info dlinfo;
  if (dladdr((const void*)(uintptr_t)installGetClassHook_untrusted, &dlinfo) == 0)
    return;
  auto machHeader = (const mach_header_platform *)dlinfo.dli_fbase;
  if (machHeader->filetype != MH_EXECUTE)
    return;
  
  // FIXME: delete this #if and dlsym once we don't
  // need to build with older libobjc headers
#if !OBJC_GETCLASSHOOK_DEFINED
  using objc_hook_getClass =  BOOL(*)(const char * _Nonnull name,
                                      Class _Nullable * _Nonnull outClass);
  auto objc_setHook_getClass =
    (void(*)(objc_hook_getClass _Nonnull,
             objc_hook_getClass _Nullable * _Nonnull))
    dlsym(RTLD_DEFAULT, "objc_setHook_getClass");
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability"
  if (objc_setHook_getClass) {
    objc_setHook_getClass(getObjCClassByMangledName_untrusted,
                          &OldGetClassHook);
  }
#pragma clang diagnostic pop
}
