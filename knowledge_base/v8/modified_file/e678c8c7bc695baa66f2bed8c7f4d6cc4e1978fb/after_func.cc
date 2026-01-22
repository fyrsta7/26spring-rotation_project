void AccessorAssembler::HandleLoadICSmiHandlerCase(
    const LoadICParameters* p, Node* holder, Node* smi_handler, Label* miss,
    ExitPoint* exit_point, bool throw_reference_error_if_nonexistent,
    ElementSupport support_elements) {
  Variable var_double_value(this, MachineRepresentation::kFloat64);
  Label rebox_double(this, &var_double_value);

  Node* handler_word = SmiUntag(smi_handler);
  Node* handler_kind = DecodeWord<LoadHandler::KindBits>(handler_word);
  if (support_elements == kSupportElements) {
    Label property(this);
    GotoIfNot(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kElement)),
              &property);

    Comment("element_load");
    Node* intptr_index = TryToIntptr(p->name, miss);
    Node* elements = LoadElements(holder);
    Node* is_jsarray_condition =
        IsSetWord<LoadHandler::IsJsArrayBits>(handler_word);
    Node* elements_kind =
        DecodeWord32FromWord<LoadHandler::ElementsKindBits>(handler_word);
    Label if_hole(this), unimplemented_elements_kind(this);
    Label* out_of_bounds = miss;
    EmitElementLoad(holder, elements, elements_kind, intptr_index,
                    is_jsarray_condition, &if_hole, &rebox_double,
                    &var_double_value, &unimplemented_elements_kind,
                    out_of_bounds, miss, exit_point);

    Bind(&unimplemented_elements_kind);
    {
      // Smi handlers should only be installed for supported elements kinds.
      // Crash if we get here.
      DebugBreak();
      Goto(miss);
    }

    Bind(&if_hole);
    {
      Comment("convert hole");
      GotoIfNot(IsSetWord<LoadHandler::ConvertHoleBits>(handler_word), miss);
      Node* protector_cell = LoadRoot(Heap::kArrayProtectorRootIndex);
      DCHECK(isolate()->heap()->array_protector()->IsPropertyCell());
      GotoIfNot(
          WordEqual(LoadObjectField(protector_cell, PropertyCell::kValueOffset),
                    SmiConstant(Smi::FromInt(Isolate::kProtectorValid))),
          miss);
      exit_point->Return(UndefinedConstant());
    }

    Bind(&property);
    Comment("property_load");
  }

  Label constant(this), field(this), normal(this, Label::kDeferred),
      interceptor(this, Label::kDeferred), nonexistent(this),
      accessor(this, Label::kDeferred), global(this, Label::kDeferred);
  GotoIf(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kField)), &field);

  GotoIf(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kConstant)),
         &constant);

  GotoIf(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kNonExistent)),
         &nonexistent);

  GotoIf(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kNormal)),
         &normal);

  GotoIf(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kAccessor)),
         &accessor);

  Branch(WordEqual(handler_kind, IntPtrConstant(LoadHandler::kGlobal)), &global,
         &interceptor);

  Bind(&field);
  HandleLoadField(holder, handler_word, &var_double_value, &rebox_double,
                  exit_point);

  Bind(&nonexistent);
  // This is a handler for a load of a non-existent value.
  if (throw_reference_error_if_nonexistent) {
    exit_point->ReturnCallRuntime(Runtime::kThrowReferenceError, p->context,
                                  p->name);
  } else {
    exit_point->Return(UndefinedConstant());
  }

  Bind(&constant);
  {
    Comment("constant_load");
    Node* descriptors = LoadMapDescriptors(LoadMap(holder));
    Node* descriptor = DecodeWord<LoadHandler::DescriptorBits>(handler_word);
    Node* scaled_descriptor =
        IntPtrMul(descriptor, IntPtrConstant(DescriptorArray::kEntrySize));
    Node* value_index =
        IntPtrAdd(scaled_descriptor,
                  IntPtrConstant(DescriptorArray::kFirstIndex +
                                 DescriptorArray::kEntryValueIndex));
    CSA_ASSERT(this,
               UintPtrLessThan(descriptor,
                               LoadAndUntagFixedArrayBaseLength(descriptors)));
    Node* value = LoadFixedArrayElement(descriptors, value_index);

    Label if_accessor_info(this, Label::kDeferred);
    GotoIf(IsSetWord<LoadHandler::IsAccessorInfoBits>(handler_word),
           &if_accessor_info);
    exit_point->Return(value);

    Bind(&if_accessor_info);
    Callable callable = CodeFactory::ApiGetter(isolate());
    exit_point->ReturnCallStub(callable, p->context, p->receiver, holder,
                               value);
  }

  Bind(&normal);
  {
    Comment("load_normal");
    Node* properties = LoadProperties(holder);
    Variable var_name_index(this, MachineType::PointerRepresentation());
    Label found(this, &var_name_index);
    NameDictionaryLookup<NameDictionary>(properties, p->name, &found,
                                         &var_name_index, miss);
    Bind(&found);
    {
      Variable var_details(this, MachineRepresentation::kWord32);
      Variable var_value(this, MachineRepresentation::kTagged);
      LoadPropertyFromNameDictionary(properties, var_name_index.value(),
                                     &var_details, &var_value);
      Node* value = CallGetterIfAccessor(var_value.value(), var_details.value(),
                                         p->context, p->receiver, miss);
      exit_point->Return(value);
    }
  }

  Bind(&accessor);
  {
    Comment("accessor_load");
    Node* descriptors = LoadMapDescriptors(LoadMap(holder));
    Node* descriptor = DecodeWord<LoadHandler::DescriptorBits>(handler_word);
    Node* scaled_descriptor =
        IntPtrMul(descriptor, IntPtrConstant(DescriptorArray::kEntrySize));
    Node* value_index =
        IntPtrAdd(scaled_descriptor,
                  IntPtrConstant(DescriptorArray::kFirstIndex +
                                 DescriptorArray::kEntryValueIndex));
    CSA_ASSERT(this,
               UintPtrLessThan(descriptor,
                               LoadAndUntagFixedArrayBaseLength(descriptors)));
    Node* accessor_pair = LoadFixedArrayElement(descriptors, value_index);
    CSA_ASSERT(this, IsAccessorPair(accessor_pair));
    Node* getter = LoadObjectField(accessor_pair, AccessorPair::kGetterOffset);
    CSA_ASSERT(this, Word32BinaryNot(IsTheHole(getter)));

    Callable callable = CodeFactory::Call(isolate());
    exit_point->Return(CallJS(callable, p->context, getter, p->receiver));
  }

  Bind(&global);
  {
    CSA_ASSERT(this, IsPropertyCell(holder));
    // Ensure the property cell doesn't contain the hole.
    Node* value = LoadObjectField(holder, PropertyCell::kValueOffset);
    Node* details =
        LoadAndUntagToWord32ObjectField(holder, PropertyCell::kDetailsOffset);
    GotoIf(IsTheHole(value), miss);

    exit_point->Return(
        CallGetterIfAccessor(value, details, p->context, p->receiver, miss));
  }

  Bind(&interceptor);
  {
    Comment("load_interceptor");
    exit_point->ReturnCallRuntime(Runtime::kLoadPropertyWithInterceptor,
                                  p->context, p->name, p->receiver, holder,
                                  p->slot, p->vector);
  }

  Bind(&rebox_double);
  exit_point->Return(AllocateHeapNumberWithValue(var_double_value.value()));
}
