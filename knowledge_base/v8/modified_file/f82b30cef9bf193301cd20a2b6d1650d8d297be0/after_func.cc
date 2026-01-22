bool AddressSpaceReservation::Allocate(void* address, size_t size,
                                       OS::MemoryPermission access) {
  // The region is already mmap'ed, so it just has to be made accessible now.
  DCHECK(Contains(address, size));
  if (access == OS::MemoryPermission::kNoAccess) {
    // Nothing to do. We don't want to call SetPermissions with kNoAccess here
    // as that will for example mark the pages as discardable, which is
    // probably not desired here.
    return true;
  }
  return OS::SetPermissions(address, size, access);
}
