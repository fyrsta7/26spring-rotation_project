IGNITION_HANDLER(TestTypeOf, InterpreterAssembler) {
  Node* object = GetAccumulator();
  Node* literal_flag = BytecodeOperandFlag(0);

#define MAKE_LABEL(name, lower_case) Label if_##lower_case(this);
  TYPEOF_LITERAL_LIST(MAKE_LABEL)
#undef MAKE_LABEL

#define LABEL_POINTER(name, lower_case) &if_##lower_case,
  Label* labels[] = {TYPEOF_LITERAL_LIST(LABEL_POINTER)};
#undef LABEL_POINTER

#define CASE(name, lower_case) \
  static_cast<int32_t>(TestTypeOfFlags::LiteralFlag::k##name),
  int32_t cases[] = {TYPEOF_LITERAL_LIST(CASE)};
#undef CASE

  Label if_true(this), if_false(this), end(this);

  // We juse use the final label as the default and properly CSA_ASSERT
  // that the {literal_flag} is valid here; this significantly improves
  // the generated code (compared to having a default label that aborts).
  unsigned const num_cases = arraysize(cases);
  CSA_ASSERT(this, Uint32LessThan(literal_flag, Int32Constant(num_cases)));
  Switch(literal_flag, labels[num_cases - 1], cases, labels, num_cases - 1);

  BIND(&if_number);
  {
    Comment("IfNumber");
    GotoIfNumber(object, &if_true);
    Goto(&if_false);
  }
  BIND(&if_string);
  {
    Comment("IfString");
    GotoIf(TaggedIsSmi(object), &if_false);
    Branch(IsString(object), &if_true, &if_false);
  }
  BIND(&if_symbol);
  {
    Comment("IfSymbol");
    GotoIf(TaggedIsSmi(object), &if_false);
    Branch(IsSymbol(object), &if_true, &if_false);
  }
  BIND(&if_boolean);
  {
    Comment("IfBoolean");
    GotoIf(WordEqual(object, BooleanConstant(true)), &if_true);
    Branch(WordEqual(object, BooleanConstant(false)), &if_true, &if_false);
  }
  BIND(&if_undefined);
  {
    Comment("IfUndefined");
    GotoIf(TaggedIsSmi(object), &if_false);
    // Check it is not null and the map has the undetectable bit set.
    GotoIf(IsNull(object), &if_false);
    Branch(IsUndetectableMap(LoadMap(object)), &if_true, &if_false);
  }
  BIND(&if_function);
  {
    Comment("IfFunction");
    GotoIf(TaggedIsSmi(object), &if_false);
    // Check if callable bit is set and not undetectable.
    Node* map_bitfield = LoadMapBitField(LoadMap(object));
    Node* callable_undetectable = Word32And(
        map_bitfield,
        Int32Constant(1 << Map::kIsUndetectable | 1 << Map::kIsCallable));
    Branch(Word32Equal(callable_undetectable,
                       Int32Constant(1 << Map::kIsCallable)),
           &if_true, &if_false);
  }
  BIND(&if_object);
  {
    Comment("IfObject");
    GotoIf(TaggedIsSmi(object), &if_false);

    // If the object is null then return true.
    GotoIf(WordEqual(object, NullConstant()), &if_true);

    // Check if the object is a receiver type and is not undefined or callable.
    Node* map = LoadMap(object);
    GotoIfNot(IsJSReceiverMap(map), &if_false);
    Node* map_bitfield = LoadMapBitField(map);
    Node* callable_undetectable = Word32And(
        map_bitfield,
        Int32Constant(1 << Map::kIsUndetectable | 1 << Map::kIsCallable));
    Branch(Word32Equal(callable_undetectable, Int32Constant(0)), &if_true,
           &if_false);
  }
  BIND(&if_other);
  {
    // Typeof doesn't return any other string value.
    Goto(&if_false);
  }

  BIND(&if_false);
  {
    SetAccumulator(BooleanConstant(false));
    Goto(&end);
  }
  BIND(&if_true);
  {
    SetAccumulator(BooleanConstant(true));
    Goto(&end);
  }
  BIND(&end);
  Dispatch();
}
