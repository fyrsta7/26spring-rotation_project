bool Compiler::CompileOptimized(Handle<JSFunction> function,
                                ConcurrencyMode mode, CodeKind code_kind) {
  DCHECK(CodeKindIsOptimizedJSFunction(code_kind));

  Isolate* isolate = function->GetIsolate();
  DCHECK(AllowCompilation::IsAllowed(isolate));

  Handle<Code> code;
  if (!GetOptimizedCode(function, mode, code_kind).ToHandle(&code)) {
    // Optimization failed, get the existing code. We could have optimized code
    // from a lower tier here. Unoptimized code must exist already if we are
    // optimizing.
    DCHECK(!isolate->has_pending_exception());
    DCHECK(function->shared().is_compiled());
    DCHECK(function->shared().IsInterpreted());
    code = ContinuationForConcurrentOptimization(isolate, function);
  }

  if (!CodeKindIsNativeContextIndependentJSFunction(code_kind)) {
    function->set_code(*code);
  }

  // Check postconditions on success.
  DCHECK(!isolate->has_pending_exception());
  DCHECK(function->shared().is_compiled());
  DCHECK(CodeKindIsNativeContextIndependentJSFunction(code_kind) ||
         function->is_compiled());
  if (!CodeKindIsNativeContextIndependentJSFunction(code_kind)) {
    DCHECK_IMPLIES(function->HasOptimizationMarker(),
                   function->IsInOptimizationQueue());
    DCHECK_IMPLIES(function->HasOptimizationMarker(),
                   function->ChecksOptimizationMarker());
    DCHECK_IMPLIES(function->IsInOptimizationQueue(),
                   mode == ConcurrencyMode::kConcurrent);
  }
  return true;
}
