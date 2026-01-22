void IteratorBuiltinsAssembler::FastIterableToList(
    TNode<Context> context, TNode<Object> iterable,
    TVariable<Object>* var_result, Label* slow) {
  Label done(this), check_string(this), check_map(this), check_set(this);

  GotoIfNot(IsFastJSArrayWithNoCustomIteration(iterable, context),
            &check_string);

  // Fast path for fast JSArray.
  *var_result =
      CallBuiltin(Builtins::kCloneFastJSArrayFillingHoles, context, iterable);
  Goto(&done);

  BIND(&check_string);
  {
    Label string_maybe_fast_call(this);
    StringBuiltinsAssembler string_assembler(state());
    string_assembler.BranchIfStringPrimitiveWithNoCustomIteration(
        iterable, context, &string_maybe_fast_call, &check_map);

    BIND(&string_maybe_fast_call);
    TNode<IntPtrT> const length = LoadStringLengthAsWord(CAST(iterable));
    // Use string length as conservative approximation of number of codepoints.
    GotoIf(
        IntPtrGreaterThan(length, IntPtrConstant(JSArray::kMaxFastArrayLength)),
        slow);
    *var_result = CallBuiltin(Builtins::kStringToList, context, iterable);
    Goto(&done);
  }

  BIND(&check_map);
  {
    Label map_fast_call(this);
    BranchIfIterableWithOriginalKeyOrValueMapIterator(
        state(), iterable, context, &map_fast_call, &check_set);

    BIND(&map_fast_call);
    *var_result = CallBuiltin(Builtins::kMapIteratorToList, context, iterable);
    Goto(&done);
  }

  BIND(&check_set);
  {
    Label set_fast_call(this);
    BranchIfIterableWithOriginalValueSetIterator(state(), iterable, context,
                                                 &set_fast_call, slow);

    BIND(&set_fast_call);
    *var_result =
        CallBuiltin(Builtins::kSetOrSetIteratorToList, context, iterable);
    Goto(&done);
  }

  BIND(&done);
}
