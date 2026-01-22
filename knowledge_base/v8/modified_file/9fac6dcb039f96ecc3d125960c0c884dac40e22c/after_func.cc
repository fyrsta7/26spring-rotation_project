Reduction JSCallReducer::ReduceArrayReduce(Handle<JSFunction> function,
                                           Node* node,
                                           ArrayReduceDirection direction) {
  if (!FLAG_turbo_inline_array_builtins) return NoChange();
  DCHECK_EQ(IrOpcode::kJSCall, node->opcode());
  CallParameters const& p = CallParametersOf(node->op());
  if (p.speculation_mode() == SpeculationMode::kDisallowSpeculation) {
    return NoChange();
  }
  bool left = direction == ArrayReduceDirection::kArrayReduceLeft;

  Node* outer_frame_state = NodeProperties::GetFrameStateInput(node);
  Node* effect = NodeProperties::GetEffectInput(node);
  Node* control = NodeProperties::GetControlInput(node);
  Node* context = NodeProperties::GetContextInput(node);

  // Try to determine the {receiver} map.
  Node* receiver = NodeProperties::GetValueInput(node, 1);
  Node* fncallback = node->op()->ValueInputCount() > 2
                         ? NodeProperties::GetValueInput(node, 2)
                         : jsgraph()->UndefinedConstant();

  ZoneHandleSet<Map> receiver_maps;
  NodeProperties::InferReceiverMapsResult result =
      NodeProperties::InferReceiverMaps(receiver, effect, &receiver_maps);
  if (result == NodeProperties::kNoReceiverMaps) return NoChange();

  ElementsKind kind = IsDoubleElementsKind(receiver_maps[0]->elements_kind())
                          ? PACKED_DOUBLE_ELEMENTS
                          : PACKED_ELEMENTS;
  for (Handle<Map> receiver_map : receiver_maps) {
    ElementsKind next_kind = receiver_map->elements_kind();
    if (!CanInlineArrayIteratingBuiltin(receiver_map)) return NoChange();
    if (!IsFastElementsKind(next_kind) || next_kind == HOLEY_DOUBLE_ELEMENTS) {
      return NoChange();
    }
    if (IsDoubleElementsKind(kind) != IsDoubleElementsKind(next_kind)) {
      return NoChange();
    }
    if (IsHoleyElementsKind(next_kind)) {
      kind = HOLEY_ELEMENTS;
    }
  }

  // Install code dependencies on the {receiver} prototype maps and the
  // global array protector cell.
  dependencies()->AssumePropertyCell(factory()->no_elements_protector());

  // If we have unreliable maps, we need a map check.
  if (result == NodeProperties::kUnreliableReceiverMaps) {
    effect =
        graph()->NewNode(simplified()->CheckMaps(CheckMapsFlag::kNone,
                                                 receiver_maps, p.feedback()),
                         receiver, effect, control);
  }

  Node* original_length = effect = graph()->NewNode(
      simplified()->LoadField(AccessBuilder::ForJSArrayLength(PACKED_ELEMENTS)),
      receiver, effect, control);

  Node* initial_index =
      left ? jsgraph()->ZeroConstant()
           : graph()->NewNode(simplified()->NumberSubtract(), original_length,
                              jsgraph()->OneConstant());
  const Operator* next_op =
      left ? simplified()->NumberAdd() : simplified()->NumberSubtract();
  Node* k = initial_index;

  std::vector<Node*> checkpoint_params({receiver, fncallback, k,
                                        original_length,
                                        jsgraph()->UndefinedConstant()});
  const int stack_parameters = static_cast<int>(checkpoint_params.size());

  Builtins::Name builtin_lazy =
      left ? Builtins::kArrayReduceLoopLazyDeoptContinuation
           : Builtins::kArrayReduceRightLoopLazyDeoptContinuation;

  Builtins::Name builtin_eager =
      left ? Builtins::kArrayReduceLoopEagerDeoptContinuation
           : Builtins::kArrayReduceRightLoopEagerDeoptContinuation;

  // Check whether the given callback function is callable. Note that
  // this has to happen outside the loop to make sure we also throw on
  // empty arrays.
  Node* check_frame_state = CreateJavaScriptBuiltinContinuationFrameState(
      jsgraph(), function, builtin_lazy, node->InputAt(0), context,
      &checkpoint_params[0], stack_parameters - 1, outer_frame_state,
      ContinuationFrameStateMode::LAZY);
  Node* check_fail = nullptr;
  Node* check_throw = nullptr;
  WireInCallbackIsCallableCheck(fncallback, context, check_frame_state, effect,
                                &control, &check_fail, &check_throw);

  // Set initial accumulator value
  Node* cur = jsgraph()->TheHoleConstant();

  Node* initial_element_frame_state =
      CreateJavaScriptBuiltinContinuationFrameState(
          jsgraph(), function, builtin_eager, node->InputAt(0), context,
          &checkpoint_params[0], stack_parameters, outer_frame_state,
          ContinuationFrameStateMode::EAGER);

  if (node->op()->ValueInputCount() > 3) {
    cur = NodeProperties::GetValueInput(node, 3);
  } else {
    // Find first/last non holey element.
    Node* vloop = k = WireInLoopStart(k, &control, &effect);
    Node* loop = control;
    Node* eloop = effect;
    effect = graph()->NewNode(common()->Checkpoint(),
                              initial_element_frame_state, effect, control);
    Node* continue_test =
        left ? graph()->NewNode(simplified()->NumberLessThan(), k,
                                original_length)
             : graph()->NewNode(simplified()->NumberLessThanOrEqual(),
                                jsgraph()->ZeroConstant(), k);
    effect = graph()->NewNode(
        simplified()->CheckIf(DeoptimizeReason::kNoInitialElement),
        continue_test, effect, control);

    cur = SafeLoadElement(kind, receiver, control, &effect, &k, p.feedback());
    Node* next_k = graph()->NewNode(next_op, k, jsgraph()->OneConstant());

    Node* hole_test = graph()->NewNode(simplified()->ReferenceEqual(), cur,
                                       jsgraph()->TheHoleConstant());
    Node* hole_branch = graph()->NewNode(common()->Branch(BranchHint::kTrue),
                                         hole_test, control);
    Node* found_el = graph()->NewNode(common()->IfFalse(), hole_branch);
    control = found_el;
    Node* is_hole = graph()->NewNode(common()->IfTrue(), hole_branch);

    WireInLoopEnd(loop, eloop, vloop, next_k, is_hole, effect);
    k = next_k;
  }

  // Start the loop.
  Node* loop = control = graph()->NewNode(common()->Loop(2), control, control);
  Node* eloop = effect =
      graph()->NewNode(common()->EffectPhi(2), effect, effect, loop);
  Node* terminate = graph()->NewNode(common()->Terminate(), eloop, loop);
  NodeProperties::MergeControlToEnd(graph(), common(), terminate);
  Node* kloop = k = graph()->NewNode(
      common()->Phi(MachineRepresentation::kTagged, 2), k, k, loop);
  Node* curloop = cur = graph()->NewNode(
      common()->Phi(MachineRepresentation::kTagged, 2), cur, cur, loop);
  checkpoint_params[2] = k;
  checkpoint_params[4] = curloop;

  control = loop;
  effect = eloop;

  Node* continue_test =
      left
          ? graph()->NewNode(simplified()->NumberLessThan(), k, original_length)
          : graph()->NewNode(simplified()->NumberLessThanOrEqual(),
                             jsgraph()->ZeroConstant(), k);

  Node* continue_branch = graph()->NewNode(common()->Branch(BranchHint::kTrue),
                                           continue_test, control);

  Node* if_true = graph()->NewNode(common()->IfTrue(), continue_branch);
  Node* if_false = graph()->NewNode(common()->IfFalse(), continue_branch);
  control = if_true;

  Node* frame_state = CreateJavaScriptBuiltinContinuationFrameState(
      jsgraph(), function, builtin_eager, node->InputAt(0), context,
      &checkpoint_params[0], stack_parameters, outer_frame_state,
      ContinuationFrameStateMode::EAGER);

  effect =
      graph()->NewNode(common()->Checkpoint(), frame_state, effect, control);

  // Make sure the map hasn't changed during the iteration
  effect = graph()->NewNode(
      simplified()->CheckMaps(CheckMapsFlag::kNone, receiver_maps), receiver,
      effect, control);

  Node* element =
      SafeLoadElement(kind, receiver, control, &effect, &k, p.feedback());

  Node* next_k = graph()->NewNode(next_op, k, jsgraph()->OneConstant());
  checkpoint_params[2] = next_k;

  Node* hole_true = nullptr;
  Node* hole_false = nullptr;
  Node* effect_true = effect;

  if (IsHoleyElementsKind(kind)) {
    // Holey elements kind require a hole check and skipping of the element in
    // the case of a hole.
    Node* check = graph()->NewNode(simplified()->ReferenceEqual(), element,
                                   jsgraph()->TheHoleConstant());
    Node* branch =
        graph()->NewNode(common()->Branch(BranchHint::kFalse), check, control);
    hole_true = graph()->NewNode(common()->IfTrue(), branch);
    hole_false = graph()->NewNode(common()->IfFalse(), branch);
    control = hole_false;

    // The contract is that we don't leak "the hole" into "user JavaScript",
    // so we must rename the {element} here to explicitly exclude "the hole"
    // from the type of {element}.
    element = effect = graph()->NewNode(
        common()->TypeGuard(Type::NonInternal()), element, effect, control);
  }

  frame_state = CreateJavaScriptBuiltinContinuationFrameState(
      jsgraph(), function, builtin_lazy, node->InputAt(0), context,
      &checkpoint_params[0], stack_parameters - 1, outer_frame_state,
      ContinuationFrameStateMode::LAZY);

  Node* next_cur = control = effect =
      graph()->NewNode(javascript()->Call(6, p.frequency()), fncallback,
                       jsgraph()->UndefinedConstant(), cur, element, k,
                       receiver, context, frame_state, effect, control);

  // Rewire potential exception edges.
  Node* on_exception = nullptr;
  if (NodeProperties::IsExceptionalCall(node, &on_exception)) {
    RewirePostCallbackExceptionEdges(check_throw, on_exception, effect,
                                     &check_fail, &control);
  }

  if (IsHoleyElementsKind(kind)) {
    Node* after_call_control = control;
    Node* after_call_effect = effect;
    control = hole_true;
    effect = effect_true;

    control = graph()->NewNode(common()->Merge(2), control, after_call_control);
    effect = graph()->NewNode(common()->EffectPhi(2), effect, after_call_effect,
                              control);
    next_cur =
        graph()->NewNode(common()->Phi(MachineRepresentation::kTagged, 2), cur,
                         next_cur, control);
  }

  k = next_k;
  cur = next_cur;

  loop->ReplaceInput(1, control);
  kloop->ReplaceInput(1, k);
  curloop->ReplaceInput(1, cur);
  eloop->ReplaceInput(1, effect);

  control = if_false;
  effect = eloop;

  // Wire up the branch for the case when IsCallable fails for the callback.
  // Since {check_throw} is an unconditional throw, it's impossible to
  // return a successful completion. Therefore, we simply connect the successful
  // completion to the graph end.
  Node* throw_node =
      graph()->NewNode(common()->Throw(), check_throw, check_fail);
  NodeProperties::MergeControlToEnd(graph(), common(), throw_node);

  ReplaceWithValue(node, curloop, effect, control);
  return Replace(curloop);
}
