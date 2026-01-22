FullObjectSlot TracedNode::Publish(Tagged<Object> object,
                                   bool needs_young_bit_update,
                                   bool needs_black_allocation,
                                   bool has_old_host, bool is_droppable_value) {
  DCHECK(FlagsAreCleared());

  if (needs_young_bit_update) {
    set_is_in_young_list(true);
  }
  if (needs_black_allocation) {
    set_markbit();
  }
  if (has_old_host) {
    DCHECK(is_in_young_list());
    set_has_old_host(true);
  }
  if (is_droppable_value) {
    set_droppable(true);
  }
  set_is_in_use(true);
  reinterpret_cast<std::atomic<Address>*>(&object_)->store(
      object.ptr(), std::memory_order_release);
  return FullObjectSlot(&object_);
}
