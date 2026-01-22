void AccessorAssembler::LoadIC_Uninitialized(const LoadICParameters* p) {
  Label miss(this);
  Node* receiver = p->receiver;
  GotoIf(TaggedIsSmi(receiver), &miss);
  Node* receiver_map = LoadMap(receiver);
  Node* instance_type = LoadMapInstanceType(receiver_map);

  // Optimistically write the state transition to the vector.
  StoreFixedArrayElement(p->vector, p->slot,
                         LoadRoot(Heap::kpremonomorphic_symbolRootIndex),
                         SKIP_WRITE_BARRIER, 0, SMI_PARAMETERS);

  {
    // Special case for Function.prototype load, because it's very common
    // for ICs that are only executed once (MyFunc.prototype.foo = ...).
    Label not_function_prototype(this);
    GotoIf(Word32NotEqual(instance_type, Int32Constant(JS_FUNCTION_TYPE)),
           &not_function_prototype);
    GotoIfNot(IsPrototypeString(p->name), &not_function_prototype);
    Node* bit_field = LoadMapBitField(receiver_map);
    GotoIf(IsSetWord32(bit_field, 1 << Map::kHasNonInstancePrototype),
           &not_function_prototype);
    Return(LoadJSFunctionPrototype(receiver, &miss));
    BIND(&not_function_prototype);
  }

  GenericPropertyLoad(receiver, receiver_map, instance_type, p->name, p, &miss,
                      kDontUseStubCache);

  BIND(&miss);
  {
    // Undo the optimistic state transition.
    StoreFixedArrayElement(p->vector, p->slot,
                           LoadRoot(Heap::kuninitialized_symbolRootIndex),
                           SKIP_WRITE_BARRIER, 0, SMI_PARAMETERS);

    TailCallRuntime(Runtime::kLoadIC_Miss, p->context, p->receiver, p->name,
                    p->slot, p->vector);
  }
}
