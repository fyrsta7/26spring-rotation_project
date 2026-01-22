void Heap::NotifyObjectSizeChange(HeapObject object, int old_size, int new_size,
                                  ClearRecordedSlots clear_recorded_slots) {
  DCHECK_LE(new_size, old_size);
  if (new_size == old_size) return;

  const bool is_background = LocalHeap::Current() != nullptr;
  DCHECK_IMPLIES(is_background,
                 clear_recorded_slots == ClearRecordedSlots::kNo);

  const ClearFreedMemoryMode clear_memory_mode =
      is_background ? ClearFreedMemoryMode::kDontClearFreedMemory
                    : ClearFreedMemoryMode::kClearFreedMemory;
  const VerifyNoSlotsRecorded verify_no_slots_recorded =
      is_background ? VerifyNoSlotsRecorded::kNo : VerifyNoSlotsRecorded::kYes;

  const Address filler = object.address() + new_size;
  const int filler_size = old_size - new_size;
  CreateFillerObjectAtRaw(filler, filler_size, clear_memory_mode,
                          clear_recorded_slots, verify_no_slots_recorded);
}
