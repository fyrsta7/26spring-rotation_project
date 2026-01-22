Reduction JSTypedLowering::ReduceJSAdd(Node* node) {
  JSBinopReduction r(this, node);
  NumberOperationHint hint;
  if (r.GetBinaryNumberOperationHint(&hint)) {
    if (hint == NumberOperationHint::kNumberOrOddball &&
        r.BothInputsAre(Type::PlainPrimitive()) &&
        r.NeitherInputCanBe(Type::StringOrReceiver())) {
      // JSAdd(x:-string, y:-string) => NumberAdd(ToNumber(x), ToNumber(y))
      r.ConvertInputsToNumber();
      return r.ChangeToPureOperator(simplified()->NumberAdd(), Type::Number());
    }
    return r.ChangeToSpeculativeOperator(
        simplified()->SpeculativeNumberAdd(hint), Type::Number());
  }
  if (r.BothInputsAre(Type::Number())) {
    // JSAdd(x:number, y:number) => NumberAdd(x, y)
    r.ConvertInputsToNumber();
    return r.ChangeToPureOperator(simplified()->NumberAdd(), Type::Number());
  }
  if ((r.BothInputsAre(Type::PlainPrimitive()) ||
       !(flags() & kDeoptimizationEnabled)) &&
      r.NeitherInputCanBe(Type::StringOrReceiver())) {
    // JSAdd(x:-string, y:-string) => NumberAdd(ToNumber(x), ToNumber(y))
    r.ConvertInputsToNumber();
    return r.ChangeToPureOperator(simplified()->NumberAdd(), Type::Number());
  }
  if (r.OneInputIs(Type::String())) {
    if (r.ShouldCreateConsString()) {
      return ReduceCreateConsString(node);
    }
    StringAddFlags flags = STRING_ADD_CHECK_NONE;
    if (!r.LeftInputIs(Type::String())) {
      flags = STRING_ADD_CONVERT_LEFT;
    } else if (!r.RightInputIs(Type::String())) {
      flags = STRING_ADD_CONVERT_RIGHT;
    }
    // JSAdd(x:string, y) => CallStub[StringAdd](x, y)
    // JSAdd(x, y:string) => CallStub[StringAdd](x, y)
    Callable const callable =
        CodeFactory::StringAdd(isolate(), flags, NOT_TENURED);
    CallDescriptor const* const desc = Linkage::GetStubCallDescriptor(
        isolate(), graph()->zone(), callable.descriptor(), 0,
        CallDescriptor::kNeedsFrameState, node->op()->properties());
    DCHECK_EQ(1, OperatorProperties::GetFrameStateInputCount(node->op()));
    node->InsertInput(graph()->zone(), 0,
                      jsgraph()->HeapConstant(callable.code()));
    NodeProperties::ChangeOp(node, common()->Call(desc));
    return Changed(node);
  }
  return NoChange();
}
