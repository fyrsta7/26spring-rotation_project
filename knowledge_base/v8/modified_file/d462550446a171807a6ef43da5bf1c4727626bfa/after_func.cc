bool HGraphBuilder::TryInlineBuiltinFunctionCall(Call* expr, bool drop_extra) {
  if (!expr->target()->shared()->HasBuiltinFunctionId()) return false;
  BuiltinFunctionId id = expr->target()->shared()->builtin_function_id();
  switch (id) {
    case kMathRound:
    case kMathFloor:
    case kMathAbs:
    case kMathSqrt:
    case kMathLog:
    case kMathSin:
    case kMathCos:
      if (expr->arguments()->length() == 1) {
        HValue* argument = Pop();
        HValue* context = environment()->LookupContext();
        Drop(1);  // Receiver.
        HUnaryMathOperation* op =
            new(zone()) HUnaryMathOperation(context, argument, id);
        op->set_position(expr->position());
        if (drop_extra) Drop(1);  // Optionally drop the function.
        ast_context()->ReturnInstruction(op, expr->id());
        return true;
      }
      break;
    default:
      // Not supported for inlining yet.
      break;
  }
  return false;
}
