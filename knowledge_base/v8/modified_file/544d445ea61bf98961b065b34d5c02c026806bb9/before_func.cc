void FullCodeGenerator::EmitIsStringWrapperSafeForDefaultValueOf(
    CallRuntime* expr) {
  ZoneList<Expression*>* args = expr->arguments();
  ASSERT(args->length() == 1);
  VisitForAccumulatorValue(args->at(0));

  Label materialize_true, materialize_false, skip_lookup;
  Label* if_true = NULL;
  Label* if_false = NULL;
  Label* fall_through = NULL;
  context()->PrepareTest(&materialize_true, &materialize_false,
                         &if_true, &if_false, &fall_through);

  Register object = x0;
  __ AssertNotSmi(object);

  Register map = x10;
  Register bitfield2 = x11;
  __ Ldr(map, FieldMemOperand(object, HeapObject::kMapOffset));
  __ Ldrb(bitfield2, FieldMemOperand(map, Map::kBitField2Offset));
  __ Tbnz(bitfield2, Map::kStringWrapperSafeForDefaultValueOf, &skip_lookup);

  // Check for fast case object. Generate false result for slow case object.
  Register props = x12;
  Register props_map = x12;
  Register hash_table_map = x13;
  __ Ldr(props, FieldMemOperand(object, JSObject::kPropertiesOffset));
  __ Ldr(props_map, FieldMemOperand(props, HeapObject::kMapOffset));
  __ LoadRoot(hash_table_map, Heap::kHashTableMapRootIndex);
  __ Cmp(props_map, hash_table_map);
  __ B(eq, if_false);

  // Look for valueOf name in the descriptor array, and indicate false if found.
  // Since we omit an enumeration index check, if it is added via a transition
  // that shares its descriptor array, this is a false positive.
  Label loop, done;

  // Skip loop if no descriptors are valid.
  Register descriptors = x12;
  Register descriptors_length = x13;
  __ NumberOfOwnDescriptors(descriptors_length, map);
  __ Cbz(descriptors_length, &done);

  __ LoadInstanceDescriptors(map, descriptors);

  // Calculate the end of the descriptor array.
  Register descriptors_end = x14;
  __ Mov(x15, DescriptorArray::kDescriptorSize);
  __ Mul(descriptors_length, descriptors_length, x15);
  // Calculate location of the first key name.
  __ Add(descriptors, descriptors,
         DescriptorArray::kFirstOffset - kHeapObjectTag);
  // Calculate the end of the descriptor array.
  __ Add(descriptors_end, descriptors,
         Operand(descriptors_length, LSL, kPointerSizeLog2));

  // Loop through all the keys in the descriptor array. If one of these is the
  // string "valueOf" the result is false.
  // TODO(all): optimise this loop to combine the add and ldr into an
  // addressing mode.
  Register valueof_string = x1;
  __ Mov(valueof_string, Operand(isolate()->factory()->value_of_string()));
  __ Bind(&loop);
  __ Ldr(x15, MemOperand(descriptors));
  __ Cmp(x15, valueof_string);
  __ B(eq, if_false);
  __ Add(descriptors, descriptors,
         DescriptorArray::kDescriptorSize * kPointerSize);
  __ Cmp(descriptors, descriptors_end);
  __ B(ne, &loop);

  __ Bind(&done);

  // Set the bit in the map to indicate that there is no local valueOf field.
  __ Ldrb(x2, FieldMemOperand(map, Map::kBitField2Offset));
  __ Orr(x2, x2, 1 << Map::kStringWrapperSafeForDefaultValueOf);
  __ Strb(x2, FieldMemOperand(map, Map::kBitField2Offset));

  __ Bind(&skip_lookup);

  // If a valueOf property is not found on the object check that its prototype
  // is the unmodified String prototype. If not result is false.
  Register prototype = x1;
  Register global_idx = x2;
  Register native_context = x2;
  Register string_proto = x3;
  Register proto_map = x4;
  __ Ldr(prototype, FieldMemOperand(map, Map::kPrototypeOffset));
  __ JumpIfSmi(prototype, if_false);
  __ Ldr(proto_map, FieldMemOperand(prototype, HeapObject::kMapOffset));
  __ Ldr(global_idx, GlobalObjectMemOperand());
  __ Ldr(native_context,
         FieldMemOperand(global_idx, GlobalObject::kNativeContextOffset));
  __ Ldr(string_proto,
         ContextMemOperand(native_context,
                           Context::STRING_FUNCTION_PROTOTYPE_MAP_INDEX));
  __ Cmp(proto_map, string_proto);

  PrepareForBailoutBeforeSplit(expr, true, if_true, if_false);
  Split(eq, if_true, if_false, fall_through);

  context()->Plug(if_true, if_false);
}
