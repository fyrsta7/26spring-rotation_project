bool SlotInterference(const VarState& a, const VarState& b) {
  return a.is_stack() && b.is_stack() &&
         b.offset() > a.offset() - value_kind_size(a.kind()) &&
         b.offset() - value_kind_size(b.kind()) < a.offset();
}
