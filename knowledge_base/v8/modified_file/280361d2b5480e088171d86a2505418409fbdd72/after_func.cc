bool LoadElimination::AliasStateInfo::MayAlias(Node* other) const {
  // If {object} is being initialized right here (indicated by {object} being
  // an Allocate node instead of a FinishRegion node), we know that {other}
  // can only alias with {object} if they refer to exactly the same node.
  if (object_->opcode() == IrOpcode::kAllocate) {
    return object_ == other;
  }
  // Decide aliasing based on the node kinds.
  if (!compiler::MayAlias(object_, other)) {
    return false;
  }
  // Decide aliasing based on maps (if available).
  Handle<Map> map;
  if (map_.ToHandle(&map)) {
    ZoneHandleSet<Map> other_maps;
    if (state_->LookupMaps(other, &other_maps) && other_maps.size() == 1) {
      if (map.address() != other_maps.at(0).address()) {
        return false;
      }
    }
  }
  return true;
}
