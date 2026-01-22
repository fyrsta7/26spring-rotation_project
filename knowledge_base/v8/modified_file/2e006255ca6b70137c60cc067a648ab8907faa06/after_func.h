namespace internal {

class V8_EXPORT MakeGarbageCollectedTraitInternal {
 protected:
  static inline void MarkObjectAsFullyConstructed(const void* payload) {
    // See api_constants for an explanation of the constants.
    std::atomic<uint16_t>* atomic_mutable_bitfield =
        reinterpret_cast<std::atomic<uint16_t>*>(
            const_cast<uint16_t*>(reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(payload) -
                api_constants::kFullyConstructedBitFieldOffsetFromPayload)));
    // It's safe to split use load+store here (instead of a read-modify-write
    // operation), since it's guaranteed that this 16-bit bitfield is only
    // modified by a single thread. This is cheaper in terms of code bloat (on
    // ARM) and performance.
    uint16_t value = atomic_mutable_bitfield->load(std::memory_order_relaxed);
    value |= api_constants::kFullyConstructedBitMask;
    atomic_mutable_bitfield->store(value, std::memory_order_release);
  }

  template <typename U, typename CustomSpace>
  struct SpacePolicy {
    static void* Allocate(AllocationHandle& handle, size_t size) {
      // Custom space.
      static_assert(std::is_base_of<CustomSpaceBase, CustomSpace>::value,
                    "Custom space must inherit from CustomSpaceBase.");
      return MakeGarbageCollectedTraitInternal::Allocate(
          handle, size, internal::GCInfoTrait<U>::Index(),
          CustomSpace::kSpaceIndex);
    }
  };

  template <typename U>
  struct SpacePolicy<U, void> {
    static void* Allocate(AllocationHandle& handle, size_t size) {
      // Default space.
      return MakeGarbageCollectedTraitInternal::Allocate(
          handle, size, internal::GCInfoTrait<U>::Index());
    }
  };

 private:
  static void* Allocate(cppgc::AllocationHandle& handle, size_t size,
                        GCInfoIndex index);
  static void* Allocate(cppgc::AllocationHandle& handle, size_t size,
                        GCInfoIndex index, CustomSpaceIndex space_index);

