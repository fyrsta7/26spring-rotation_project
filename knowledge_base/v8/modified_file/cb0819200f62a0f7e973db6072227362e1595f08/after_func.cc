void AccessorAssembler::HandlePolymorphicCase(
    Node* receiver_map, TNode<WeakFixedArray> feedback, Label* if_handler,
    TVariable<MaybeObject>* var_handler, Label* if_miss) {
  Comment("HandlePolymorphicCase");
  DCHECK_EQ(MachineRepresentation::kTagged, var_handler->rep());

  // Iterate {feedback} array.
  const int kEntrySize = 2;

  // Load the {feedback} array length.
  TNode<IntPtrT> length = LoadAndUntagWeakFixedArrayLength(feedback);
  CSA_ASSERT(this, IntPtrLessThanOrEqual(IntPtrConstant(kEntrySize), length));

  // This is a hand-crafted loop that iterates backwards and only compares
  // against zero at the end, since we already know that we will have at least a
  // single entry in the {feedback} array anyways.
  TVARIABLE(IntPtrT, var_index, IntPtrSub(length, IntPtrConstant(kEntrySize)));
  Label loop(this, &var_index), loop_next(this);
  Goto(&loop);
  BIND(&loop);
  {
    TNode<MaybeObject> maybe_cached_map =
        LoadWeakFixedArrayElement(feedback, var_index.value());
    CSA_ASSERT(this, IsWeakOrCleared(maybe_cached_map));
    GotoIf(IsNotWeakReferenceTo(maybe_cached_map, CAST(receiver_map)),
           &loop_next);

    // Found, now call handler.
    TNode<MaybeObject> handler =
        LoadWeakFixedArrayElement(feedback, var_index.value(), kTaggedSize);
    *var_handler = handler;
    Goto(if_handler);

    BIND(&loop_next);
    var_index =
        Signed(IntPtrSub(var_index.value(), IntPtrConstant(kEntrySize)));
    Branch(IntPtrGreaterThanOrEqual(var_index.value(), IntPtrConstant(0)),
           &loop, if_miss);
  }
}
