void AccessorAssembler::HandlePolymorphicCase(
    Node* receiver_map, TNode<WeakFixedArray> feedback, Label* if_handler,
    TVariable<MaybeObject>* var_handler, Label* if_miss) {
  Comment("HandlePolymorphicCase");
  DCHECK_EQ(MachineRepresentation::kTagged, var_handler->rep());

  // Iterate {feedback} array.
  const int kEntrySize = 2;

  // Load the {feedback} array length.
  TNode<IntPtrT> length = LoadAndUntagWeakFixedArrayLength(feedback);
  CSA_ASSERT(this, IntPtrLessThanOrEqual(IntPtrConstant(1), length));

  // This is a hand-crafted loop that only compares against the {length}
  // in the end, since we already know that we will have at least a single
  // entry in the {feedback} array anyways.
  TVARIABLE(IntPtrT, var_index, IntPtrConstant(0));
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
        Signed(IntPtrAdd(var_index.value(), IntPtrConstant(kEntrySize)));
    Branch(IntPtrLessThan(var_index.value(), length), &loop, if_miss);
  }
}
