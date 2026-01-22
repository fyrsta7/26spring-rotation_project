    return nullptr;
}
#endif

extern "C" {

static void* os_alloc(size_t size, const char* name)
{
    int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_PURGEABLE;
#if ARCH(X86_64)
    flags |= MAP_RANDOMIZED;
#endif
