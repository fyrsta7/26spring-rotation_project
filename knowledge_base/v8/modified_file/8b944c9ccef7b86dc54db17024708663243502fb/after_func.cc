TF_BUILTIN(ObjectAssign, ObjectBuiltinsAssembler) {
  TNode<IntPtrT> argc = ChangeInt32ToIntPtr(
      UncheckedParameter<Int32T>(Descriptor::kJSActualArgumentsCount));
  CodeStubArguments args(this, argc);

  auto context = Parameter<Context>(Descriptor::kContext);
  TNode<Object> target = args.GetOptionalArgumentValue(0);

  TVARIABLE(IntPtrT, slow_path_index, IntPtrConstant(1));

  // 1. Let to be ? ToObject(target).
  TNode<JSReceiver> to = ToObject_Inline(context, target);

  Label done(this);
  // 2. If only one argument was passed, return to.
  TNode<IntPtrT> args_length = args.GetLengthWithoutReceiver();
  GotoIf(UintPtrLessThanOrEqual(args_length, IntPtrConstant(1)), &done);

  // First let's try a fastpath specifically for when the target objects is an
  // empty object literal.
  // TODO(olivf): For the cases where we could detect that the object literal
  // does not escape in the parser already, we should have a variant of this
  // builtin where the target is not yet allocated at all.
  Label done_fast_path(this), slow_path(this);
  GotoIfForceSlowPath(&slow_path);
  {
    Label fall_through_slow_path(this);

    // First, evaluate the first source object.
    TNode<Object> source = args.GetOptionalArgumentValue(1);
    GotoIf(IsNullOrUndefined(source), &done_fast_path);

    TVARIABLE(IntPtrT, var_result_index, IntPtrConstant(0));
    TNode<JSReceiver> from = ToObject_Inline(context, source);

    TNode<Map> from_map = LoadMap(from);
    TNode<Map> to_map = LoadMap(to);

    // Chances are very slim that cloning is possible if we have different
    // instance sizes.
    // TODO(olivf): Re-Evaluate this once we have a faster target map lookup
    // that does not need to go through the runtime.
    TNode<IntPtrT> from_inst_size = LoadMapInstanceSizeInWords(from_map);
    TNode<IntPtrT> to_inst_size = LoadMapInstanceSizeInWords(to_map);
    GotoIfNot(IntPtrEqual(from_inst_size, to_inst_size), &slow_path);

    // Both source and target should be in fastmode, not a prototype and not
    // deprecated.
    constexpr uint32_t field3_exclusion_mask =
        Map::Bits3::IsDictionaryMapBit::kMask |
        Map::Bits3::IsDeprecatedBit::kMask |
        Map::Bits3::IsPrototypeMapBit::kMask;

    // Ensure the target is empty and extensible and has none of the exclusion
    // bits set.
    TNode<Uint32T> target_field3 = LoadMapBitField3(to_map);
    TNode<Uint32T> field3_descriptors_and_extensible_mask = Uint32Constant(
        Map::Bits3::NumberOfOwnDescriptorsBits::kMask |
        Map::Bits3::IsExtensibleBit::kMask | field3_exclusion_mask);
    // If the masked field3 equals the extensible bit, then the number of
    // descriptors was 0 -- which is what we need here.
    GotoIfNot(
        Word32Equal(
            Uint32Constant(Map::Bits3::IsExtensibleBit::encode(true)),
            Word32And(target_field3, field3_descriptors_and_extensible_mask)),
        &slow_path);

    // For the fastcase we want the source to be a JSObject and the target a
    // fresh empty object literal.
    TNode<NativeContext> native_context = LoadNativeContext(context);
    TNode<Map> empty_object_literal_map =
        LoadObjectFunctionInitialMap(native_context);
    GotoIfNot(TaggedEqual(to_map, empty_object_literal_map), &slow_path);
    GotoIfNot(IsJSObjectMap(from_map), &slow_path);

    // Check that the source is in fastmode, not a prototype and not deprecated.
    TNode<Uint32T> source_field3 = LoadMapBitField3(from_map);
    TNode<Uint32T> field3_exclusion_mask_const =
        Uint32Constant(field3_exclusion_mask);
    GotoIfNot(
        Word32Equal(Uint32Constant(0),
                    Word32And(source_field3, field3_exclusion_mask_const)),
        &slow_path);
    CSA_DCHECK(this, Word32BinaryNot(IsElementsKindInRange(
                         LoadElementsKind(to_map),
                         FIRST_ANY_NONEXTENSIBLE_ELEMENTS_KIND,
                         LAST_ANY_NONEXTENSIBLE_ELEMENTS_KIND)));

    // TODO(olivf): We could support the case when the `to` has elements, but
    // the source doesn't. But there is a danger of then caching an invalid
    // transition when the converse happens later.
    GotoIfNot(TaggedEqual(LoadElements(CAST(to)), EmptyFixedArrayConstant()),
              &slow_path);

    // Check if our particular source->target combination is fast clonable.
    // E.g., this ensures that we only have fast properties and in general that
    // the binary layout is compatible for `FastCloneJSObject`.
    // TODO(olivf): Add a fastcase to read the cached target map from the
    // transition array without going through the runtime.
    Label continue_fast_path(this);
    TNode<HeapObject> maybe_clone_map =
        CAST(CallRuntime(Runtime::kObjectAssignTryFastcase, context, from, to));
    GotoIf(TaggedEqual(maybe_clone_map, UndefinedConstant()), &slow_path);
    GotoIf(TaggedEqual(maybe_clone_map, TrueConstant()), &done_fast_path);
    CSA_DCHECK(this, IsMap(maybe_clone_map));
    Goto(&continue_fast_path);

    BIND(&continue_fast_path);
    TNode<Map> clone_map = CAST(maybe_clone_map);
    CSA_DCHECK(this, IntPtrEqual(LoadMapInstanceSizeInWords(to_map),
                                 LoadMapInstanceSizeInWords(clone_map)));
    CSA_DCHECK(this,
               IntPtrEqual(LoadMapInobjectPropertiesStartInWords(to_map),
                           LoadMapInobjectPropertiesStartInWords(clone_map)));
    FastCloneJSObject(
        from, from_map, clone_map,
        [&](TNode<Map> map, TNode<HeapObject> properties,
            TNode<FixedArray> elements) {
          StoreMap(to, clone_map);
          StoreJSReceiverPropertiesOrHash(to, properties);
          StoreJSObjectElements(CAST(to), elements);
          return to;
        },
        false /* target_is_new */);

    Goto(&done_fast_path);
    BIND(&done_fast_path);

    // If the fast path above succeeded we must skip assigning the first source
    // object in the generic implementation below.
    slow_path_index = IntPtrConstant(2);
    Branch(IntPtrGreaterThan(args_length, IntPtrConstant(2)), &slow_path,
           &done);
  }
  BIND(&slow_path);

  // 3. Let sources be the List of argument values starting with the
  //    second argument.
  // 4. For each element nextSource of sources, in ascending index order,
  {
    args.ForEach(
        [=](TNode<Object> next_source) {
          CallBuiltin(Builtin::kSetDataProperties, context, to, next_source);
        },
        slow_path_index.value());
    Goto(&done);
  }

  // 5. Return to.
  BIND(&done);
  args.PopAndReturn(to);
}
