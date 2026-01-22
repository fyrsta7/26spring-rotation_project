#endif
    return append_with_ip_and_bp(current_thread->pid(), current_thread->tid(), 0, base_pointer, type, 0, arg1, arg2, arg3);
}

static Vector<FlatPtr, PerformanceEvent::max_stack_frame_count> raw_backtrace(FlatPtr bp, FlatPtr ip)
{
    Vector<FlatPtr, PerformanceEvent::max_stack_frame_count> backtrace;
    if (ip != 0)
        backtrace.append(ip);
    FlatPtr stack_ptr_copy;
    FlatPtr stack_ptr = bp;
    // FIXME: Figure out how to remove this SmapDisabler without breaking profile stacks.
    SmapDisabler disabler;
    // NOTE: The stack should always have kernel frames first, followed by userspace frames.
    //       If a userspace frame points back into kernel memory, something is afoot.
    bool is_walking_userspace_stack = false;
    while (stack_ptr) {
        void* fault_at;
        if (!safe_memcpy(&stack_ptr_copy, (void*)stack_ptr, sizeof(FlatPtr), fault_at))
            break;
        if (!Memory::is_user_address(VirtualAddress { stack_ptr })) {
            if (is_walking_userspace_stack) {
                dbgln("SHENANIGANS! Userspace stack points back into kernel memory");
                break;
            }
        } else {
            is_walking_userspace_stack = true;
        }
        FlatPtr retaddr;
        if (!safe_memcpy(&retaddr, (void*)(stack_ptr + sizeof(FlatPtr)), sizeof(FlatPtr), fault_at))
            break;
        if (retaddr == 0)
            break;
        backtrace.append(retaddr);
        if (backtrace.size() == PerformanceEvent::max_stack_frame_count)
            break;
