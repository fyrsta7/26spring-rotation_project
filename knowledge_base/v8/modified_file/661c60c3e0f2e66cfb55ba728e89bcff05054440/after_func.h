FullObjectSlot TracedNode::Publish(Tagged<Object> object,
                                   bool needs_young_bit_update,
                                   bool needs_black_allocation,
                                   bool has_old_host, bool is_droppable_value) {
  DCHECK(FlagsAreCleared());

  flags_ = needs_young_bit_update << IsInYoungList::kShift |
           needs_black_allocation << Markbit::kShift |
           has_old_host << HasOldHost::kShift |
           is_droppable_value << IsDroppable::kShift | 1 << IsInUse::kShift;
  reinterpret_cast<std::atomic<Address>*>(&object_)->store(
      object.ptr(), std::memory_order_release);
  return FullObjectSlot(&object_);
}
