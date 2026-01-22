Handle<HeapObject> Constant::ToHeapObject() const {
  DCHECK_EQ(kHeapObject, type());
  Handle<HeapObject> value(
      bit_cast<HeapObject**>(static_cast<intptr_t>(value_)));
  return value;
}
