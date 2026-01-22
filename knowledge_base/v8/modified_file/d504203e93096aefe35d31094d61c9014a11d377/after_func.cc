Reduction JSTypedLowering::ReduceJSAdd(Node* node) {
  JSBinopReduction r(this, node);
  if (r.BothInputsAre(Type::Number())) {
    // JSAdd(x:number, y:number) => NumberAdd(x, y)
    return r.ChangeToPureOperator(simplified()->NumberAdd(), Type::Number());
  }
  if (r.BothInputsAre(Type::PlainPrimitive()) &&
      r.NeitherInputCanBe(Type::StringOrReceiver())) {
    // JSAdd(x:-string, y:-string) => NumberAdd(ToNumber(x), ToNumber(y))
    r.ConvertInputsToNumber();
    return r.ChangeToPureOperator(simplified()->NumberAdd(), Type::Number());
  }
  if (BinaryOperationHintOf(node->op()) == BinaryOperationHint::kString) {
    // Always bake in String feedback into the graph.
    // TODO(bmeurer): Consider adding a SpeculativeStringAdd operator,
    // and use that in JSTypeHintLowering instead of looking at the
    // binary operation feedback here.
    r.CheckInputsToString();
  }
  if (r.OneInputIs(Type::String())) {
    // We know that (at least) one input is already a String,
    // so try to strength-reduce the non-String input.
    if (r.LeftInputIs(Type::String())) {
      Reduction const reduction = ReduceJSToStringInput(r.right());
      if (reduction.Changed()) {
        NodeProperties::ReplaceValueInput(node, reduction.replacement(), 1);
      }
    } else if (r.RightInputIs(Type::String())) {
      Reduction const reduction = ReduceJSToStringInput(r.left());
      if (reduction.Changed()) {
        NodeProperties::ReplaceValueInput(node, reduction.replacement(), 0);
      }
    }
    // We might be able to constant-fold the String concatenation now.
    if (r.BothInputsAre(Type::String())) {
      HeapObjectBinopMatcher m(node);
      if (m.IsFoldable()) {
        Handle<String> left = Handle<String>::cast(m.left().Value());
        Handle<String> right = Handle<String>::cast(m.right().Value());
        if (left->length() + right->length() > String::kMaxLength) {
          // No point in trying to optimize this, as it will just throw.
          return NoChange();
        }
        Node* value = jsgraph()->HeapConstant(
            factory()->NewConsString(left, right).ToHandleChecked());
        ReplaceWithValue(node, value);
        return Replace(value);
      }
    }
    // We might know for sure that we're creating a ConsString here.
    if (r.ShouldCreateConsString()) {
      return ReduceCreateConsString(node);
    }
    // Eliminate useless concatenation of empty string.
    if (r.BothInputsAre(Type::String())) {
      Node* effect = NodeProperties::GetEffectInput(node);
      Node* control = NodeProperties::GetControlInput(node);
      if (r.LeftInputIs(empty_string_type_)) {
        Node* value = effect =
            graph()->NewNode(simplified()->CheckString(VectorSlotPair()),
                             r.right(), effect, control);
        ReplaceWithValue(node, value, effect, control);
        return Replace(value);
      } else if (r.RightInputIs(empty_string_type_)) {
        Node* value = effect =
            graph()->NewNode(simplified()->CheckString(VectorSlotPair()),
                             r.left(), effect, control);
        ReplaceWithValue(node, value, effect, control);
        return Replace(value);
      }
    }
    StringAddFlags flags = STRING_ADD_CHECK_NONE;
    if (!r.LeftInputIs(Type::String())) {
      flags = STRING_ADD_CONVERT_LEFT;
    } else if (!r.RightInputIs(Type::String())) {
      flags = STRING_ADD_CONVERT_RIGHT;
    }
    Operator::Properties properties = node->op()->properties();
    if (r.NeitherInputCanBe(Type::Receiver())) {
      // Both sides are already strings, so we know that the
      // string addition will not cause any observable side
      // effects; it can still throw obviously.
      properties = Operator::kNoWrite | Operator::kNoDeopt;
    }
    // JSAdd(x:string, y) => CallStub[StringAdd](x, y)
    // JSAdd(x, y:string) => CallStub[StringAdd](x, y)
    Callable const callable =
        CodeFactory::StringAdd(isolate(), flags, NOT_TENURED);
    auto call_descriptor = Linkage::GetStubCallDescriptor(
        isolate(), graph()->zone(), callable.descriptor(), 0,
        CallDescriptor::kNeedsFrameState, properties);
    DCHECK_EQ(1, OperatorProperties::GetFrameStateInputCount(node->op()));
    node->InsertInput(graph()->zone(), 0,
                      jsgraph()->HeapConstant(callable.code()));
    NodeProperties::ChangeOp(node, common()->Call(call_descriptor));
    return Changed(node);
  }
  // We never get here when we had String feedback.
  DCHECK_NE(BinaryOperationHint::kString, BinaryOperationHintOf(node->op()));
  return NoChange();
}
