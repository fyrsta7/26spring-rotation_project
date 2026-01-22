void KeyedStoreGenericAssembler::KeyedStoreGeneric() {
  typedef StoreWithVectorDescriptor Descriptor;

  Node* receiver = Parameter(Descriptor::kReceiver);
  Node* name = Parameter(Descriptor::kName);
  Node* value = Parameter(Descriptor::kValue);
  Node* slot = Parameter(Descriptor::kSlot);
  Node* vector = Parameter(Descriptor::kVector);
  Node* context = Parameter(Descriptor::kContext);

  VARIABLE(var_index, MachineType::PointerRepresentation());
  VARIABLE(var_unique, MachineRepresentation::kTagged);
  var_unique.Bind(name);  // Dummy initialization.
  Label if_index(this), if_unique_name(this), slow(this);

  GotoIf(TaggedIsSmi(receiver), &slow);
  Node* receiver_map = LoadMap(receiver);
  Node* instance_type = LoadMapInstanceType(receiver_map);
  // Receivers requiring non-standard element accesses (interceptors, access
  // checks, strings and string wrappers, proxies) are handled in the runtime.
  GotoIf(Int32LessThanOrEqual(instance_type,
                              Int32Constant(LAST_CUSTOM_ELEMENTS_RECEIVER)),
         &slow);

  TryToName(name, &if_index, &var_index, &if_unique_name, &var_unique, &slow);

  BIND(&if_index);
  {
    Comment("integer index");
    EmitGenericElementStore(receiver, receiver_map, instance_type,
                            var_index.value(), value, context, &slow);
  }

  BIND(&if_unique_name);
  {
    Comment("key is unique name");
    StoreICParameters p(context, receiver, var_unique.value(), value, slot,
                        vector);
    EmitGenericPropertyStore(receiver, receiver_map, &p, &slow);
  }

  BIND(&slow);
  {
    Comment("KeyedStoreGeneric_slow");
    VARIABLE(var_language_mode, MachineRepresentation::kTaggedSigned,
             SmiConstant(LanguageMode::kStrict));
    Label call_runtime(this);
    BranchIfStrictMode(vector, slot, &call_runtime);
    var_language_mode.Bind(SmiConstant(LanguageMode::kSloppy));
    Goto(&call_runtime);
    BIND(&call_runtime);
    TailCallRuntime(Runtime::kSetProperty, context, receiver, name, value,
                    var_language_mode.value());
  }
}
