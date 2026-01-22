  class_ = node_class;
}

void Node::Clear() {
  in_edges_.clear();
  out_edges_.clear();
  id_ = -1;
  cost_id_ = -1;
  class_ = NC_UNINITIALIZED;
  props_.reset();
  assigned_device_name_index_ = 0;
}
