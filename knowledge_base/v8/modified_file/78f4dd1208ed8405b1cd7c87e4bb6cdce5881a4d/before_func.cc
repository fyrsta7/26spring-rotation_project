void HGraphBuilder::VisitSub(UnaryOperation* expr) {
  CHECK_ALIVE(VisitForValue(expr->expression()));
  HValue* value = Pop();
  HInstruction* instr = new(zone()) HMul(value, graph_->GetConstantMinus1());
  ast_context()->ReturnInstruction(instr, expr->id());
}
