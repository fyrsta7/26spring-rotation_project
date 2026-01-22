static bool IsOptimizable(JSFunction* function) {
  if (Heap::InNewSpace(function)) return false;
  Code* code = function->code();
  return code->kind() == Code::FUNCTION && code->optimizable();
}
