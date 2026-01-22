template <AllocationType type>
V8_WARN_UNUSED_RESULT V8_INLINE AllocationResult HeapAllocator::AllocateRaw(
    int size_in_bytes, AllocationOrigin origin, AllocationAlignment alignment) {
  DCHECK_EQ(heap_->gc_state(), Heap::NOT_IN_GC);
  DCHECK(AllowHandleAllocation::IsAllowed());
  DCHECK(AllowHeapAllocation::IsAllowed());

  if (FLAG_single_generation && type == AllocationType::kYoung) {
    return AllocateRaw(size_in_bytes, AllocationType::kOld, origin, alignment);
  }

#ifdef V8_ENABLE_ALLOCATION_TIMEOUT
  if (FLAG_random_gc_interval > 0 || FLAG_gc_interval >= 0) {
    if (!heap_->always_allocate() && allocation_timeout_-- <= 0) {
      return AllocationResult::Failure();
    }
  }
#endif  // V8_ENABLE_ALLOCATION_TIMEOUT

#ifdef DEBUG
  IncrementObjectCounters();
#endif  // DEBUG

  if (heap_->CanSafepoint()) {
    heap_->main_thread_local_heap()->Safepoint();
  }

  const size_t large_object_threshold = heap_->MaxRegularHeapObjectSize(type);
  const bool large_object =
      static_cast<size_t>(size_in_bytes) > large_object_threshold;

  HeapObject object;
  AllocationResult allocation;

  if (V8_ENABLE_THIRD_PARTY_HEAP_BOOL) {
    allocation = heap_->tp_heap_->Allocate(size_in_bytes, type, alignment);
  } else {
    if (V8_UNLIKELY(large_object)) {
      allocation =
          AllocateRawLargeInternal(size_in_bytes, type, origin, alignment);
    } else {
      switch (type) {
        case AllocationType::kYoung:
          allocation =
              new_space()->AllocateRaw(size_in_bytes, alignment, origin);
          break;
        case AllocationType::kOld:
          allocation =
              old_space()->AllocateRaw(size_in_bytes, alignment, origin);
          break;
        case AllocationType::kCode:
          DCHECK_EQ(alignment, AllocationAlignment::kTaggedAligned);
          DCHECK(AllowCodeAllocation::IsAllowed());
          allocation = code_space()->AllocateRaw(
              size_in_bytes, AllocationAlignment::kTaggedAligned);
          break;
        case AllocationType::kMap:
          DCHECK_EQ(alignment, AllocationAlignment::kTaggedAligned);
          allocation = space_for_maps()->AllocateRaw(
              size_in_bytes, AllocationAlignment::kTaggedAligned);
          break;
        case AllocationType::kReadOnly:
          DCHECK(read_only_space()->writable());
          DCHECK_EQ(AllocationOrigin::kRuntime, origin);
          allocation = read_only_space()->AllocateRaw(size_in_bytes, alignment);
          break;
        case AllocationType::kSharedMap:
          allocation = shared_map_allocator_->AllocateRaw(size_in_bytes,
                                                          alignment, origin);
          break;
        case AllocationType::kSharedOld:
          allocation = shared_old_allocator_->AllocateRaw(size_in_bytes,
                                                          alignment, origin);
          break;
      }
    }
  }

  if (allocation.To(&object)) {
    if (AllocationType::kCode == type && !V8_ENABLE_THIRD_PARTY_HEAP_BOOL) {
      // Unprotect the memory chunk of the object if it was not unprotected
      // already.
      heap_->UnprotectAndRegisterMemoryChunk(
          object, UnprotectMemoryOrigin::kMainThread);
      heap_->ZapCodeObject(object.address(), size_in_bytes);
      if (!large_object) {
        MemoryChunk::FromHeapObject(object)
            ->GetCodeObjectRegistry()
            ->RegisterNewlyAllocatedCodeObject(object.address());
      }
    }

#ifdef V8_ENABLE_CONSERVATIVE_STACK_SCANNING
    if (AllocationType::kReadOnly != type) {
      DCHECK_TAG_ALIGNED(object.address());
      Page::FromHeapObject(object)->object_start_bitmap()->SetBit(
          object.address());
    }
#endif  // V8_ENABLE_CONSERVATIVE_STACK_SCANNING

    for (auto& tracker : heap_->allocation_trackers_) {
      tracker->AllocationEvent(object.address(), size_in_bytes);
    }
  }

  return allocation;
}
