void HGraphBuilder::VisitSub(UnaryOperation* expr) {
  CHECK_ALIVE(VisitForValue(expr->expression()));
  HValue* value = Pop();
  HInstruction* instr = new(zone()) HMul(value, graph_->GetConstantMinus1());
  TypeInfo info = oracle()->UnaryType(expr);
  Representation rep = ToRepresentation(info);
  TraceRepresentation(expr->op(), info, instr, rep);
  AssumeRepresentation(instr, rep);
  ast_context()->ReturnInstruction(instr, expr->id());
}
