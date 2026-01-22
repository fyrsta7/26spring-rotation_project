bool Node::OwnedBy(Node const* owner) const {
  for (Use* use = first_use_; use; use = use->next) {
    if (use->from() != owner) {
      return false;
    }
  }
  return first_use_ != nullptr;
}
