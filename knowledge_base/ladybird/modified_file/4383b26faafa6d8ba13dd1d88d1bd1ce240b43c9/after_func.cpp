    VERIFY(!framebuffer_device_or_error.is_error());
    return framebuffer_device_or_error.release_value();
}

ErrorOr<Memory::Region*> FramebufferDevice::mmap(Process& process, OpenFileDescription&, Memory::VirtualRange const& range, u64 offset, int prot, bool shared)
{
    TRY(process.require_promise(Pledge::video));
    SpinlockLocker lock(m_activation_lock);
    if (!shared)
        return ENODEV;
    if (offset != 0)
        return ENXIO;
    auto framebuffer_length = TRY(buffer_length(0));
    framebuffer_length = TRY(Memory::page_round_up(framebuffer_length));
    if (range.size() != framebuffer_length)
        return EOVERFLOW;

    m_userspace_real_framebuffer_vmobject = TRY(Memory::AnonymousVMObject::try_create_for_physical_range(m_framebuffer_address, framebuffer_length));
    m_real_framebuffer_vmobject = TRY(Memory::AnonymousVMObject::try_create_for_physical_range(m_framebuffer_address, framebuffer_length));
    m_swapped_framebuffer_vmobject = TRY(Memory::AnonymousVMObject::try_create_with_size(framebuffer_length, AllocationStrategy::AllocateNow));
    m_real_framebuffer_region = TRY(MM.allocate_kernel_region_with_vmobject(*m_real_framebuffer_vmobject, framebuffer_length, "Framebuffer", Memory::Region::Access::ReadWrite));
    m_swapped_framebuffer_region = TRY(MM.allocate_kernel_region_with_vmobject(*m_swapped_framebuffer_vmobject, framebuffer_length, "Framebuffer Swap (Blank)", Memory::Region::Access::ReadWrite));

    RefPtr<Memory::VMObject> chosen_vmobject;
    if (m_graphical_writes_enabled) {
        chosen_vmobject = m_real_framebuffer_vmobject;
    } else {
        chosen_vmobject = m_swapped_framebuffer_vmobject;
    }
    m_userspace_framebuffer_region = TRY(process.address_space().allocate_region_with_vmobject(
        range,
        chosen_vmobject.release_nonnull(),
        0,
        "Framebuffer",
        prot,
        shared));
