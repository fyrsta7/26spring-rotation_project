void* DynamicMessage::MutableRaw(int i) {
  return OffsetToPointer(
      OffsetValue(type_info_->offsets[i], type_info_->type->field(i)->type()));
}
