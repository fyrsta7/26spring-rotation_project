void KeyedStoreGenericAssembler::EmitGenericPropertyStore(
    Node* receiver, Node* receiver_map, const StoreICParameters* p, Label* slow,
    LanguageMode language_mode) {
  Variable var_accessor_pair(this, MachineRepresentation::kTagged);
  Variable var_accessor_holder(this, MachineRepresentation::kTagged);
  Label stub_cache(this), fast_properties(this), dictionary_properties(this),
      accessor(this), readonly(this);
  Node* properties = LoadProperties(receiver);
  Node* properties_map = LoadMap(properties);
  Branch(WordEqual(properties_map, LoadRoot(Heap::kHashTableMapRootIndex)),
         &dictionary_properties, &fast_properties);

  Bind(&fast_properties);
  {
    // TODO(jkummerow): Does it make sense to support some cases here inline?
    // Maybe overwrite existing writable properties?
    // Maybe support map transitions?
    Goto(&stub_cache);
  }

  Bind(&dictionary_properties);
  {
    Comment("dictionary property store");
    // We checked for LAST_CUSTOM_ELEMENTS_RECEIVER before, which rules out
    // seeing global objects here (which would need special handling).

    Variable var_name_index(this, MachineType::PointerRepresentation());
    Label dictionary_found(this, &var_name_index), not_found(this);
    NameDictionaryLookup<NameDictionary>(properties, p->name, &dictionary_found,
                                         &var_name_index, &not_found);
    Bind(&dictionary_found);
    {
      Label overwrite(this);
      const int kNameToDetailsOffset = (NameDictionary::kEntryDetailsIndex -
                                        NameDictionary::kEntryKeyIndex) *
                                       kPointerSize;
      Node* details = LoadAndUntagToWord32FixedArrayElement(
          properties, var_name_index.value(), kNameToDetailsOffset);
      JumpIfDataProperty(details, &overwrite, &readonly);

      // Accessor case.
      const int kNameToValueOffset =
          (NameDictionary::kEntryValueIndex - NameDictionary::kEntryKeyIndex) *
          kPointerSize;
      var_accessor_pair.Bind(LoadFixedArrayElement(
          properties, var_name_index.value(), kNameToValueOffset));
      var_accessor_holder.Bind(receiver);
      Goto(&accessor);

      Bind(&overwrite);
      {
        StoreFixedArrayElement(properties, var_name_index.value(), p->value,
                               UPDATE_WRITE_BARRIER, kNameToValueOffset);
        Return(p->value);
      }
    }

    Bind(&not_found);
    {
      LookupPropertyOnPrototypeChain(receiver_map, p->name, &accessor,
                                     &var_accessor_pair, &var_accessor_holder,
                                     &readonly, slow);
      Add<NameDictionary>(properties, p->name, p->value, slow);
      Return(p->value);
    }
  }

  Bind(&accessor);
  {
    Label not_callable(this);
    Node* accessor_pair = var_accessor_pair.value();
    GotoIf(IsAccessorInfoMap(LoadMap(accessor_pair)), slow);
    CSA_ASSERT(this, HasInstanceType(accessor_pair, ACCESSOR_PAIR_TYPE));
    Node* setter = LoadObjectField(accessor_pair, AccessorPair::kSetterOffset);
    Node* setter_map = LoadMap(setter);
    // FunctionTemplateInfo setters are not supported yet.
    GotoIf(IsFunctionTemplateInfoMap(setter_map), slow);
    GotoUnless(IsCallableMap(setter_map), &not_callable);

    Callable callable = CodeFactory::Call(isolate());
    CallJS(callable, p->context, setter, receiver, p->value);
    Return(p->value);

    Bind(&not_callable);
    {
      if (language_mode == STRICT) {
        Node* message =
            SmiConstant(Smi::FromInt(MessageTemplate::kNoSetterInCallback));
        TailCallRuntime(Runtime::kThrowTypeError, p->context, message, p->name,
                        var_accessor_holder.value());
      } else {
        DCHECK_EQ(SLOPPY, language_mode);
        Return(p->value);
      }
    }
  }

  Bind(&readonly);
  {
    if (language_mode == STRICT) {
      Node* message =
          SmiConstant(Smi::FromInt(MessageTemplate::kStrictReadOnlyProperty));
      Node* type = Typeof(p->receiver, p->context);
      TailCallRuntime(Runtime::kThrowTypeError, p->context, message, p->name,
                      type, p->receiver);
    } else {
      DCHECK_EQ(SLOPPY, language_mode);
      Return(p->value);
    }
  }

  Bind(&stub_cache);
  {
    Comment("stub cache probe");
    // The stub cache lookup is opportunistic: if we find a handler, use it;
    // otherwise take the slow path. Since this is a generic stub, compiling
    // a handler (as KeyedStoreIC_Miss would do) is probably a waste of time.
    Variable var_handler(this, MachineRepresentation::kTagged);
    Label found_handler(this, &var_handler);
    TryProbeStubCache(isolate()->store_stub_cache(), receiver, p->name,
                      &found_handler, &var_handler, slow);
    Bind(&found_handler);
    {
      Comment("KeyedStoreGeneric found handler");
      HandleStoreICHandlerCase(p, var_handler.value(), slow);
    }
  }
}
