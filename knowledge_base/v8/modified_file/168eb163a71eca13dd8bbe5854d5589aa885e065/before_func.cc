Reduction JSCallReducer::ReduceFunctionPrototypeApply(Node* node) {
  DCHECK_EQ(IrOpcode::kJSCall, node->opcode());
  Node* target = NodeProperties::GetValueInput(node, 0);
  CallParameters const& p = CallParametersOf(node->op());
  // Tail calls to Function.prototype.apply are not properly supported
  // down the pipeline, so we disable this optimization completely for
  // tail calls (for now).
  if (p.tail_call_mode() == TailCallMode::kAllow) return NoChange();
  Handle<JSFunction> apply =
      Handle<JSFunction>::cast(HeapObjectMatcher(target).Value());
  size_t arity = p.arity();
  DCHECK_LE(2u, arity);
  ConvertReceiverMode convert_mode = ConvertReceiverMode::kAny;
  if (arity == 2) {
    // Neither thisArg nor argArray was provided.
    convert_mode = ConvertReceiverMode::kNullOrUndefined;
    node->ReplaceInput(0, node->InputAt(1));
    node->ReplaceInput(1, jsgraph()->UndefinedConstant());
  } else if (arity == 3) {
    // The argArray was not provided, just remove the {target}.
    node->RemoveInput(0);
    --arity;
  } else if (arity == 4) {
    // Check if argArray is an arguments object, and {node} is the only value
    // user of argArray (except for value uses in frame states).
    Node* arg_array = NodeProperties::GetValueInput(node, 3);
    if (arg_array->opcode() != IrOpcode::kJSCreateArguments) return NoChange();
    for (Edge edge : arg_array->use_edges()) {
      if (edge.from()->opcode() == IrOpcode::kStateValues) continue;
      if (!NodeProperties::IsValueEdge(edge)) continue;
      if (edge.from() == node) continue;
      return NoChange();
    }
    // Check if the arguments can be handled in the fast case (i.e. we don't
    // have aliased sloppy arguments), and compute the {start_index} for
    // rest parameters.
    CreateArgumentsType const type = CreateArgumentsTypeOf(arg_array->op());
    Node* frame_state = NodeProperties::GetFrameStateInput(arg_array);
    FrameStateInfo state_info = OpParameter<FrameStateInfo>(frame_state);
    int start_index = 0;
    // Determine the formal parameter count;
    Handle<SharedFunctionInfo> shared;
    if (!state_info.shared_info().ToHandle(&shared)) return NoChange();
    int formal_parameter_count = shared->internal_formal_parameter_count();
    if (type == CreateArgumentsType::kMappedArguments) {
      // Mapped arguments (sloppy mode) that are aliased can only be handled
      // here if there's no side-effect between the {node} and the {arg_array}.
      // TODO(turbofan): Further relax this constraint.
      if (formal_parameter_count != 0) {
        Node* effect = NodeProperties::GetEffectInput(node);
        while (effect != arg_array) {
          if (effect->op()->EffectInputCount() != 1 ||
              !(effect->op()->properties() & Operator::kNoWrite)) {
            return NoChange();
          }
          effect = NodeProperties::GetEffectInput(effect);
        }
      }
    } else if (type == CreateArgumentsType::kRestParameter) {
      start_index = formal_parameter_count;
    }
    // Check if are applying to inlined arguments or to the arguments of
    // the outermost function.
    Node* outer_state = frame_state->InputAt(kFrameStateOuterStateInput);
    if (outer_state->opcode() != IrOpcode::kFrameState) {
      // Reduce {node} to a JSCallForwardVarargs operation, which just
      // re-pushes the incoming arguments and calls the {target}.
      node->RemoveInput(0);  // Function.prototype.apply
      node->RemoveInput(2);  // arguments
      NodeProperties::ChangeOp(node, javascript()->CallForwardVarargs(
                                         start_index, p.tail_call_mode()));
      return Changed(node);
    }
    // Get to the actual frame state from which to extract the arguments;
    // we can only optimize this in case the {node} was already inlined into
    // some other function (and same for the {arg_array}).
    FrameStateInfo outer_info = OpParameter<FrameStateInfo>(outer_state);
    if (outer_info.type() == FrameStateType::kArgumentsAdaptor) {
      // Need to take the parameters from the arguments adaptor.
      frame_state = outer_state;
    }
    // Remove the argArray input from the {node}.
    node->RemoveInput(static_cast<int>(--arity));
    // Add the actual parameters to the {node}, skipping the receiver,
    // starting from {start_index}.
    Node* const parameters = frame_state->InputAt(kFrameStateParametersInput);
    for (int i = start_index + 1; i < parameters->InputCount(); ++i) {
      node->InsertInput(graph()->zone(), static_cast<int>(arity),
                        parameters->InputAt(i));
      ++arity;
    }
    // Drop the {target} from the {node}.
    node->RemoveInput(0);
    --arity;
  } else {
    return NoChange();
  }
  // Change {node} to the new {JSCall} operator.
  NodeProperties::ChangeOp(
      node,
      javascript()->Call(arity, p.frequency(), VectorSlotPair(), convert_mode,
                         p.tail_call_mode()));
  // Change context of {node} to the Function.prototype.apply context,
  // to ensure any exception is thrown in the correct context.
  NodeProperties::ReplaceContextInput(
      node, jsgraph()->HeapConstant(handle(apply->context(), isolate())));
  // Try to further reduce the JSCall {node}.
  Reduction const reduction = ReduceJSCall(node);
  return reduction.Changed() ? reduction : Changed(node);
}
