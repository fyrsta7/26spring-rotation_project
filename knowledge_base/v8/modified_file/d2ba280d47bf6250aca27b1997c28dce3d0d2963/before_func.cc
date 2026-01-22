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
