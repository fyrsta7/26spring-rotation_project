
    return interpreter_load_result;
}

KResult Process::do_exec(NonnullRefPtr<OpenFileDescription> main_program_description, NonnullOwnPtrVector<KString> arguments, NonnullOwnPtrVector<KString> environment,
    RefPtr<OpenFileDescription> interpreter_description, Thread*& new_main_thread, u32& prev_flags, const ElfW(Ehdr) & main_program_header)
{
    VERIFY(is_user_process());
    VERIFY(!Processor::in_critical());
    auto path = TRY(main_program_description->try_serialize_absolute_path());

    dbgln_if(EXEC_DEBUG, "do_exec: {}", path);

    // FIXME: How much stack space does process startup need?
    if (!validate_stack_size(arguments, environment))
        return E2BIG;

    // FIXME: split_view() currently allocates (Vector) without checking for failure.
    auto parts = path->view().split_view('/');
    if (parts.is_empty())
        return ENOENT;

    auto new_process_name = TRY(KString::try_create(parts.last()));
    auto new_main_thread_name = TRY(new_process_name->try_clone());

    auto load_result = TRY(load(main_program_description, interpreter_description, main_program_header));

    // NOTE: We don't need the interpreter executable description after this point.
    //       We destroy it here to prevent it from getting destroyed when we return from this function.
    //       That's important because when we're returning from this function, we're in a very delicate
    //       state where we can't block (e.g by trying to acquire a mutex in description teardown.)
    bool has_interpreter = interpreter_description;
    interpreter_description = nullptr;

    auto signal_trampoline_range = TRY(load_result.space->try_allocate_range({}, PAGE_SIZE));
    auto signal_trampoline_region = TRY(load_result.space->allocate_region_with_vmobject(signal_trampoline_range, g_signal_trampoline_region->vmobject(), 0, "Signal trampoline", PROT_READ | PROT_EXEC, true));
    signal_trampoline_region->set_syscall_region(true);

    // (For dynamically linked executable) Allocate an FD for passing the main executable to the dynamic loader.
    Optional<ScopedDescriptionAllocation> main_program_fd_allocation;
    if (has_interpreter)
        main_program_fd_allocation = TRY(m_fds.allocate());

    // We commit to the new executable at this point. There is no turning back!

    // Prevent other processes from attaching to us with ptrace while we're doing this.
    MutexLocker ptrace_locker(ptrace_lock());

    // Disable profiling temporarily in case it's running on this process.
    auto was_profiling = m_profiling;
    TemporaryChange profiling_disabler(m_profiling, false);

    kill_threads_except_self();

    bool executable_is_setid = false;

    if (!(main_program_description->custody()->mount_flags() & MS_NOSUID)) {
        auto main_program_metadata = main_program_description->metadata();
        if (main_program_metadata.is_setuid()) {
            executable_is_setid = true;
            ProtectedDataMutationScope scope { *this };
            m_protected_values.euid = main_program_metadata.uid;
            m_protected_values.suid = main_program_metadata.uid;
        }
        if (main_program_metadata.is_setgid()) {
            executable_is_setid = true;
            ProtectedDataMutationScope scope { *this };
            m_protected_values.egid = main_program_metadata.gid;
            m_protected_values.sgid = main_program_metadata.gid;
        }
    }

    set_dumpable(!executable_is_setid);

    {
        // We must disable global profiling (especially kfree tracing) here because
        // we might otherwise end up walking the stack into the process' space that
        // is about to be destroyed.
        TemporaryChange global_profiling_disabler(g_profiling_all_threads, false);
        m_space = load_result.space.release_nonnull();
    }
    Memory::MemoryManager::enter_address_space(*m_space);

    m_executable = main_program_description->custody();
    m_arguments = move(arguments);
    m_environment = move(environment);

    m_veil_state = VeilState::None;
    m_unveiled_paths.clear();
    m_unveiled_paths.set_metadata({ "/", UnveilAccess::None, false });

    for (auto& property : m_coredump_properties)
        property = {};

    auto current_thread = Thread::current();
    current_thread->clear_signals();

    clear_futex_queues_on_exec();

    fds().change_each([&](auto& file_description_metadata) {
        if (file_description_metadata.is_valid() && file_description_metadata.flags() & FD_CLOEXEC)
            file_description_metadata = {};
    });

    if (main_program_fd_allocation.has_value()) {
        main_program_description->set_readable(true);
        m_fds[main_program_fd_allocation->fd].set(move(main_program_description), FD_CLOEXEC);
    }

    new_main_thread = nullptr;
    if (&current_thread->process() == this) {
        new_main_thread = current_thread;
    } else {
        for_each_thread([&](auto& thread) {
            new_main_thread = &thread;
            return IterationDecision::Break;
        });
    }
    VERIFY(new_main_thread);

    auto auxv = generate_auxiliary_vector(load_result.load_base, load_result.entry_eip, uid(), euid(), gid(), egid(), path->view(), main_program_fd_allocation);

    // NOTE: We create the new stack before disabling interrupts since it will zero-fault
    //       and we don't want to deal with faults after this point.
    auto make_stack_result = make_userspace_context_for_main_thread(new_main_thread->regs(), *load_result.stack_region.unsafe_ptr(), m_arguments, m_environment, move(auxv));
    if (make_stack_result.is_error())
        return make_stack_result.error();
    FlatPtr new_userspace_sp = make_stack_result.value();

    if (wait_for_tracer_at_next_execve()) {
        // Make sure we release the ptrace lock here or the tracer will block forever.
        ptrace_locker.unlock();
        Thread::current()->send_urgent_signal_to_self(SIGSTOP);
    } else {
        // Unlock regardless before disabling interrupts.
        // Ensure we always unlock after checking ptrace status to avoid TOCTOU ptrace issues
        ptrace_locker.unlock();
    }

    // We enter a critical section here because we don't want to get interrupted between do_exec()
    // and Processor::assume_context() or the next context switch.
    // If we used an InterruptDisabler that sti()'d on exit, we might timer tick'd too soon in exec().
    Processor::enter_critical();
    prev_flags = cpu_flags();
    cli();

    // NOTE: Be careful to not trigger any page faults below!

    m_name = move(new_process_name);
    new_main_thread->set_name(move(new_main_thread_name));

    {
        ProtectedDataMutationScope scope { *this };
        m_protected_values.promises = m_protected_values.execpromises.load();
        m_protected_values.has_promises = m_protected_values.has_execpromises.load();

        m_protected_values.execpromises = 0;
        m_protected_values.has_execpromises = false;

        m_protected_values.signal_trampoline = signal_trampoline_region->vaddr();

        // FIXME: PID/TID ISSUE
        m_protected_values.pid = new_main_thread->tid().value();
    }

    auto tsr_result = new_main_thread->make_thread_specific_region({});
    if (tsr_result.is_error()) {
        // FIXME: We cannot fail this late. Refactor this so the allocation happens before we commit to the new executable.
        VERIFY_NOT_REACHED();
    }
    new_main_thread->reset_fpu_state();

    auto& regs = new_main_thread->m_regs;
#if ARCH(I386)
    regs.cs = GDT_SELECTOR_CODE3 | 3;
    regs.ds = GDT_SELECTOR_DATA3 | 3;
    regs.es = GDT_SELECTOR_DATA3 | 3;
    regs.ss = GDT_SELECTOR_DATA3 | 3;
    regs.fs = GDT_SELECTOR_DATA3 | 3;
    regs.gs = GDT_SELECTOR_TLS | 3;
    regs.eip = load_result.entry_eip;
    regs.esp = new_userspace_sp;
#else
    regs.rip = load_result.entry_eip;
    regs.rsp = new_userspace_sp;
#endif
    regs.cr3 = address_space().page_directory().cr3();

    {
        TemporaryChange profiling_disabler(m_profiling, was_profiling);
        PerformanceManager::add_process_exec_event(*this);
    }

    {
        SpinlockLocker lock(g_scheduler_lock);
        new_main_thread->set_state(Thread::State::Runnable);
    }
    u32 lock_count_to_restore;
    [[maybe_unused]] auto rc = big_lock().force_unlock_if_locked(lock_count_to_restore);
