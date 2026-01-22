void AccessorAssembler::ExtendPropertiesBackingStore(Node* object) {
  Node* properties = LoadProperties(object);
  Node* length = LoadFixedArrayBaseLength(properties);

  ParameterMode mode = OptimalParameterMode();
  length = TaggedToParameter(length, mode);

  Node* delta = IntPtrOrSmiConstant(JSObject::kFieldsAdded, mode);
  Node* new_capacity = IntPtrOrSmiAdd(length, delta, mode);

  // Grow properties array.
  ElementsKind kind = FAST_ELEMENTS;
  DCHECK(kMaxNumberOfDescriptors + JSObject::kFieldsAdded <
         FixedArrayBase::GetMaxLengthForNewSpaceAllocation(kind));
  // The size of a new properties backing store is guaranteed to be small
  // enough that the new backing store will be allocated in new space.
  CSA_ASSERT(this,
             UintPtrOrSmiLessThan(
                 new_capacity,
                 IntPtrOrSmiConstant(
                     kMaxNumberOfDescriptors + JSObject::kFieldsAdded, mode),
                 mode));

  Node* new_properties = AllocateFixedArray(kind, new_capacity, mode);

  FillFixedArrayWithValue(kind, new_properties, length, new_capacity,
                          Heap::kUndefinedValueRootIndex, mode);

  // |new_properties| is guaranteed to be in new space, so we can skip
  // the write barrier.
  CopyFixedArrayElements(kind, properties, new_properties, length,
                         SKIP_WRITE_BARRIER, mode);

  StoreObjectField(object, JSObject::kPropertiesOffset, new_properties);
}
