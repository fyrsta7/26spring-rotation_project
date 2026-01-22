        ASSERT_NOT_REACHED();
        return 0;
    }

    if (function == SC_fork)
        return process.sys$fork(regs);

    if (function == SC_sigreturn)
        return process.sys$sigreturn(regs);

    if (function >= Function::__Count) {
        dbg() << process << ": Unknown syscall %u requested (" << arg1 << ", " << arg2 << ", " << arg3 << ")";
        return -ENOSYS;
    }

    if (s_syscall_table[function] == nullptr) {
        dbg() << process << ": Null syscall " << function << " requested: \"" << to_string((Function)function) << "\", you probably need to rebuild this program.";
        return -ENOSYS;
    }
    return (process.*(s_syscall_table[function]))(arg1, arg2, arg3);
}

}

void syscall_handler(RegisterDump regs)
{
    // Make sure SMAP protection is enabled on syscall entry.
    clac();

    // Apply a random offset in the range 0-255 to the stack pointer,
    // to make kernel stacks a bit less deterministic.
    auto* ptr = (char*)__builtin_alloca(get_fast_random<u8>());
    asm volatile(""
                 : "=m"(*ptr));

    auto& process = current->process();

    if (!MM.validate_user_stack(process, VirtualAddress(regs.userspace_esp))) {
        dbgprintf("Invalid stack pointer: %p\n", regs.userspace_esp);
        handle_crash(regs, "Bad stack on syscall entry", SIGSTKFLT);
        ASSERT_NOT_REACHED();
    }

    auto* calling_region = MM.region_from_vaddr(process, VirtualAddress(regs.eip));
    if (!calling_region) {
        dbgprintf("Syscall from %p which has no region\n", regs.eip);
        handle_crash(regs, "Syscall from unknown region", SIGSEGV);
        ASSERT_NOT_REACHED();
