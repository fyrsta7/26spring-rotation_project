HAllocate* HGraphBuilder::JSArrayBuilder::AllocateArray(
    HValue* capacity,
    HConstant* capacity_upper_bound,
    HValue* length_field,
    FillMode fill_mode) {
  return AllocateArray(capacity,
                       capacity_upper_bound->GetInteger32Constant(),
                       length_field,
                       fill_mode);
}
