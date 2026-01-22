bool AddressSpaceReservation::Allocate(void* address, size_t size,
                                       OS::MemoryPermission access) {
  // The region is already mmap'ed, so it just has to be made accessible now.
  DCHECK(Contains(address, size));
  return OS::SetPermissions(address, size, access);
}
