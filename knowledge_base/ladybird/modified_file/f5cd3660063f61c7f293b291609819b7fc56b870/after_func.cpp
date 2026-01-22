#include <Kernel/Heap/kmalloc.h>
#include <Kernel/StdLib.h>
#include <Kernel/VM/MemoryManager.h>

String copy_string_from_user(const char* user_str, size_t user_str_size)
{
    bool is_user = Kernel::is_user_range(VirtualAddress(user_str), user_str_size);
    if (!is_user)
        return {};
    Kernel::SmapDisabler disabler;
    void* fault_at;
    ssize_t length = Kernel::safe_strnlen(user_str, user_str_size, fault_at);
    if (length < 0) {
        dbgln("copy_string_from_user({:p}, {}) failed at {} (strnlen)", static_cast<const void*>(user_str), user_str_size, VirtualAddress { fault_at });
        return {};
    }
    if (length == 0)
        return String::empty();

    char* buffer;
    auto copied_string = StringImpl::create_uninitialized((size_t)length, buffer);
    if (!Kernel::safe_memcpy(buffer, user_str, (size_t)length, fault_at)) {
        dbgln("copy_string_from_user({:p}, {}) failed at {} (memcpy)", static_cast<const void*>(user_str), user_str_size, VirtualAddress { fault_at });
