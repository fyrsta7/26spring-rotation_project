void TaskCodeGenLLVM::visit(Block *stmt_list) {
  for (auto &stmt : stmt_list->statements) {
    stmt->accept(this);
    if (returned) {
      break;
    }
  }
}
