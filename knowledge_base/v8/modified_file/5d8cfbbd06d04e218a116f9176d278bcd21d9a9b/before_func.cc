Handle<HeapObject> Constant::ToHeapObject() const {
  DCHECK_EQ(kHeapObject, type());
  Handle<HeapObject> value(
      bit_cast<HeapObject**>(static_cast<intptr_t>(value_)));
  if (value->IsConsString()) {
    value = String::Flatten(Handle<String>::cast(value), TENURED);
  }
  return value;
}
