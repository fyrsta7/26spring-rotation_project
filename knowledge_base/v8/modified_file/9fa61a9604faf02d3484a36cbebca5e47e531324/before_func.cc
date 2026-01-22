bool Node::OwnedBy(Node const* owner) const {
  unsigned mask = 0;
  for (Use* use = first_use_; use; use = use->next) {
    if (use->from() == owner) {
      mask |= 1;
    } else {
      return false;
    }
  }
  return mask == 1;
}
