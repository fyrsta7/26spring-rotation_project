Reduction JSCallReducer::ReduceCallApiFunction(
    Node* node, const SharedFunctionInfoRef& shared) {
  DCHECK_EQ(IrOpcode::kJSCall, node->opcode());
  CallParameters const& p = CallParametersOf(node->op());
  int const argc = static_cast<int>(p.arity()) - 2;
  Node* target = NodeProperties::GetValueInput(node, 0);
  Node* global_proxy =
      jsgraph()->Constant(native_context().global_proxy_object());
  Node* receiver = (p.convert_mode() == ConvertReceiverMode::kNullOrUndefined)
                       ? global_proxy
                       : NodeProperties::GetValueInput(node, 1);
  Node* holder;
  Node* effect = NodeProperties::GetEffectInput(node);
  Node* control = NodeProperties::GetControlInput(node);
  Node* context = NodeProperties::GetContextInput(node);
  Node* frame_state = NodeProperties::GetFrameStateInput(node);

  // See if we can optimize this API call to {shared}.
  Handle<FunctionTemplateInfo> function_template_info(
      FunctionTemplateInfo::cast(shared.object()->function_data()), isolate());
  CallOptimization call_optimization(isolate(), function_template_info);
  if (!call_optimization.is_simple_api_call()) return NoChange();

  // Try to infer the {receiver} maps from the graph.
  MapInference inference(broker(), receiver, effect);
  if (inference.HaveMaps()) {
    MapHandles const& receiver_maps = inference.GetMaps();

    // Check that all {receiver_maps} are actually JSReceiver maps and
    // that the {function_template_info} accepts them without access
    // checks (even if "access check needed" is set for {receiver}).
    //
    // Note that we don't need to know the concrete {receiver} maps here,
    // meaning it's fine if the {receiver_maps} are unreliable, and we also
    // don't need to install any stability dependencies, since the only
    // relevant information regarding the {receiver} is the Map::constructor
    // field on the root map (which is different from the JavaScript exposed
    // "constructor" property) and that field cannot change.
    //
    // So if we know that {receiver} had a certain constructor at some point
    // in the past (i.e. it had a certain map), then this constructor is going
    // to be the same later, since this information cannot change with map
    // transitions.
    //
    // The same is true for the instance type, e.g. we still know that the
    // instance type is JSObject even if that information is unreliable, and
    // the "access check needed" bit, which also cannot change later.
    for (Handle<Map> map : receiver_maps) {
      MapRef receiver_map(broker(), map);
      if (!receiver_map.IsJSReceiverMap() ||
          (receiver_map.is_access_check_needed() &&
           !function_template_info->accept_any_receiver())) {
        return inference.NoChange();
      }
    }

    // See if we can constant-fold the compatible receiver checks.
    CallOptimization::HolderLookup lookup;
    Handle<JSObject> api_holder =
        call_optimization.LookupHolderOfExpectedType(receiver_maps[0], &lookup);
    if (lookup == CallOptimization::kHolderNotFound)
      return inference.NoChange();
    for (size_t i = 1; i < receiver_maps.size(); ++i) {
      CallOptimization::HolderLookup lookupi;
      Handle<JSObject> holderi = call_optimization.LookupHolderOfExpectedType(
          receiver_maps[i], &lookupi);
      if (lookup != lookupi) return inference.NoChange();
      if (!api_holder.is_identical_to(holderi)) return inference.NoChange();
    }

    // We may need to check {receiver_maps} again below, so better
    // make sure we are allowed to speculate in this case.
    if (p.speculation_mode() == SpeculationMode::kDisallowSpeculation) {
      return inference.NoChange();
    }

    // TODO(neis): The maps were used in a way that does not actually require
    // map checks or stability dependencies.
    inference.RelyOnMapsPreferStability(dependencies(), jsgraph(), &effect,
                                        control, p.feedback());

    // Determine the appropriate holder for the {lookup}.
    holder = lookup == CallOptimization::kHolderFound
                 ? jsgraph()->HeapConstant(api_holder)
                 : receiver;
  } else if (function_template_info->accept_any_receiver() &&
             function_template_info->signature()->IsUndefined(isolate())) {
    // We haven't found any {receiver_maps}, but we might still be able to
    // optimize the API call depending on the {function_template_info}.
    // If the API function accepts any kind of {receiver}, we only need to
    // ensure that the {receiver} is actually a JSReceiver at this point,
    // and also pass that as the {holder}. There are two independent bits
    // here:
    //
    //  a. When the "accept any receiver" bit is set, it means we don't
    //     need to perform access checks, even if the {receiver}'s map
    //     has the "needs access check" bit set.
    //  b. When the {function_template_info} has no signature, we don't
    //     need to do the compatible receiver check, since all receivers
    //     are considered compatible at that point, and the {receiver}
    //     will be pass as the {holder}.
    //
    receiver = holder = effect =
        graph()->NewNode(simplified()->ConvertReceiver(p.convert_mode()),
                         receiver, global_proxy, effect, control);
  } else {
    // We don't have enough information to eliminate the access check
    // and/or the compatible receiver check, so use the generic builtin
    // that does those checks dynamically. This is still significantly
    // faster than the generic call sequence.
    Builtins::Name builtin_name =
        !function_template_info->accept_any_receiver()
            ? (function_template_info->signature()->IsUndefined(isolate())
                   ? Builtins::kCallFunctionTemplate_CheckAccess
                   : Builtins::
                         kCallFunctionTemplate_CheckAccessAndCompatibleReceiver)
            : Builtins::kCallFunctionTemplate_CheckCompatibleReceiver;

    // The CallFunctionTemplate builtin requires the {receiver} to be
    // an actual JSReceiver, so make sure we do the proper conversion
    // first if necessary.
    receiver = holder = effect =
        graph()->NewNode(simplified()->ConvertReceiver(p.convert_mode()),
                         receiver, global_proxy, effect, control);

    Callable callable = Builtins::CallableFor(isolate(), builtin_name);
    auto call_descriptor = Linkage::GetStubCallDescriptor(
        graph()->zone(), callable.descriptor(),
        argc + 1 /* implicit receiver */, CallDescriptor::kNeedsFrameState);
    node->InsertInput(graph()->zone(), 0,
                      jsgraph()->HeapConstant(callable.code()));
    node->ReplaceInput(1, jsgraph()->HeapConstant(function_template_info));
    node->InsertInput(graph()->zone(), 2, jsgraph()->Constant(argc));
    node->ReplaceInput(3, receiver);       // Update receiver input.
    node->ReplaceInput(6 + argc, effect);  // Update effect input.
    NodeProperties::ChangeOp(node, common()->Call(call_descriptor));
    return Changed(node);
  }

  // TODO(turbofan): Consider introducing a JSCallApiCallback operator for
  // this and lower it during JSGenericLowering, and unify this with the
  // JSNativeContextSpecialization::InlineApiCall method a bit.
  Handle<CallHandlerInfo> call_handler_info(
      CallHandlerInfo::cast(function_template_info->call_code()), isolate());
  Handle<Object> data(call_handler_info->data(), isolate());
  Callable call_api_callback = CodeFactory::CallApiCallback(isolate());
  CallInterfaceDescriptor cid = call_api_callback.descriptor();
  auto call_descriptor = Linkage::GetStubCallDescriptor(
      graph()->zone(), cid, argc + 1 /* implicit receiver */,
      CallDescriptor::kNeedsFrameState);
  ApiFunction api_function(v8::ToCData<Address>(call_handler_info->callback()));
  ExternalReference function_reference = ExternalReference::Create(
      &api_function, ExternalReference::DIRECT_API_CALL);

  Node* continuation_frame_state = CreateGenericLazyDeoptContinuationFrameState(
      jsgraph(), shared, target, context, receiver, frame_state);

  node->InsertInput(graph()->zone(), 0,
                    jsgraph()->HeapConstant(call_api_callback.code()));
  node->ReplaceInput(1, jsgraph()->ExternalConstant(function_reference));
  node->InsertInput(graph()->zone(), 2, jsgraph()->Constant(argc));
  node->InsertInput(graph()->zone(), 3, jsgraph()->Constant(data));
  node->InsertInput(graph()->zone(), 4, holder);
  node->ReplaceInput(5, receiver);       // Update receiver input.
  node->ReplaceInput(7 + argc, continuation_frame_state);
  node->ReplaceInput(8 + argc, effect);  // Update effect input.
  NodeProperties::ChangeOp(node, common()->Call(call_descriptor));
  return Changed(node);
}
