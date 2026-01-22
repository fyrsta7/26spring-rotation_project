MaybeHandle<Code> GetBaselineCode(Handle<JSFunction> function) {
  Isolate* isolate = function->GetIsolate();
  VMState<COMPILER> state(isolate);
  PostponeInterruptsScope postpone(isolate);
  CompilationInfoWithZone info(function);

  // Reset profiler ticks, function is no longer considered hot.
  if (function->shared()->HasBytecodeArray()) {
    function->shared()->set_profiler_ticks(0);
  }

  // Nothing left to do if the function already has baseline code.
  if (function->shared()->code()->kind() == Code::FUNCTION) {
    return Handle<Code>(function->shared()->code());
  }

  // We do not switch to baseline code when the debugger might have created a
  // copy of the bytecode with break slots to be able to set break points.
  if (function->shared()->HasDebugInfo()) {
    return MaybeHandle<Code>();
  }

  // TODO(4280): For now we do not switch generators to baseline code because
  // there might be suspended activations stored in generator objects on the
  // heap. We could eventually go directly to TurboFan in this case.
  if (function->shared()->is_generator()) {
    return MaybeHandle<Code>();
  }

  // TODO(4280): For now we disable switching to baseline code in the presence
  // of interpreter activations of the given function. The reasons are:
  //  1) The debugger assumes each function is either full-code or bytecode.
  //  2) The underlying bytecode is cleared below, breaking stack unwinding.
  if (HasInterpreterActivations(isolate, function->shared())) {
    if (FLAG_trace_opt) {
      OFStream os(stdout);
      os << "[unable to switch " << Brief(*function) << " due to activations]"
         << std::endl;
    }
    return MaybeHandle<Code>();
  }

  if (FLAG_trace_opt) {
    OFStream os(stdout);
    os << "[switching method " << Brief(*function) << " to baseline code]"
       << std::endl;
  }

  // Parse and update CompilationInfo with the results.
  if (!Parser::ParseStatic(info.parse_info())) return MaybeHandle<Code>();
  Handle<SharedFunctionInfo> shared = info.shared_info();
  DCHECK_EQ(shared->language_mode(), info.literal()->language_mode());

  // Compile baseline code using the full code generator.
  if (!Compiler::Analyze(info.parse_info()) ||
      !FullCodeGenerator::MakeCode(&info)) {
    if (!isolate->has_pending_exception()) isolate->StackOverflow();
    return MaybeHandle<Code>();
  }

  // TODO(4280): For now we play it safe and remove the bytecode array when we
  // switch to baseline code. We might consider keeping around the bytecode so
  // that it can be used as the "source of truth" eventually.
  shared->ClearBytecodeArray();

  // Update the shared function info with the scope info.
  InstallSharedScopeInfo(&info, shared);

  // Install compilation result on the shared function info
  InstallSharedCompilationResult(&info, shared);

  // Record the function compilation event.
  RecordFunctionCompilation(Logger::LAZY_COMPILE_TAG, &info);

  return info.code();
}
