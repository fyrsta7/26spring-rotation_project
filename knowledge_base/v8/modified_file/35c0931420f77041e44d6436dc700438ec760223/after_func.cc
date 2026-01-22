bool Compiler::Compile(Isolate* isolate, Handle<SharedFunctionInfo> shared_info,
                       ClearExceptionFlag flag,
                       IsCompiledScope* is_compiled_scope,
                       CreateSourcePositions create_source_positions_flag) {
  // We should never reach here if the function is already compiled.
  DCHECK(!shared_info->is_compiled());
  DCHECK(!is_compiled_scope->is_compiled());
  DCHECK(AllowCompilation::IsAllowed(isolate));
  DCHECK_EQ(ThreadId::Current(), isolate->thread_id());
  DCHECK(!isolate->has_pending_exception());
  DCHECK(!shared_info->HasBytecodeArray());

  VMState<BYTECODE_COMPILER> state(isolate);
  PostponeInterruptsScope postpone(isolate);
  TimerEventScope<TimerEventCompileCode> compile_timer(isolate);
  RCS_SCOPE(isolate, RuntimeCallCounterId::kCompileFunction);
  TRACE_EVENT0(TRACE_DISABLED_BY_DEFAULT("v8.compile"), "V8.CompileCode");
  AggregatedHistogramTimerScope timer(isolate->counters()->compile_lazy());

  Handle<Script> script(Script::cast(shared_info->script()), isolate);

  // Set up parse info.
  UnoptimizedCompileFlags flags =
      UnoptimizedCompileFlags::ForFunctionCompile(isolate, *shared_info);
  if (create_source_positions_flag == CreateSourcePositions::kYes) {
    flags.set_collect_source_positions(true);
  }

  UnoptimizedCompileState compile_state;
  ReusableUnoptimizedCompileState reusable_state(isolate);
  ParseInfo parse_info(isolate, flags, &compile_state, &reusable_state);

  // Check if the compiler dispatcher has shared_info enqueued for compile.
  LazyCompileDispatcher* dispatcher = isolate->lazy_compile_dispatcher();
  if (dispatcher && dispatcher->IsEnqueued(shared_info)) {
    if (!dispatcher->FinishNow(shared_info)) {
      return FailWithPendingException(isolate, script, &parse_info, flag);
    }
    *is_compiled_scope = shared_info->is_compiled_scope(isolate);
    DCHECK(is_compiled_scope->is_compiled());
    return true;
  }

  if (shared_info->HasUncompiledDataWithPreparseData()) {
    parse_info.set_consumed_preparse_data(ConsumedPreparseData::For(
        isolate,
        handle(
            shared_info->uncompiled_data_with_preparse_data().preparse_data(),
            isolate)));
  }

  // Parse and update ParseInfo with the results.
  if (!parsing::ParseAny(&parse_info, shared_info, isolate,
                         parsing::ReportStatisticsMode::kYes)) {
    return FailWithPendingException(isolate, script, &parse_info, flag);
  }

  // Generate the unoptimized bytecode or asm-js data.
  FinalizeUnoptimizedCompilationDataList
      finalize_unoptimized_compilation_data_list;

  if (!IterativelyExecuteAndFinalizeUnoptimizedCompilationJobs(
          isolate, shared_info, script, &parse_info, isolate->allocator(),
          is_compiled_scope, &finalize_unoptimized_compilation_data_list,
          nullptr)) {
    return FailWithPendingException(isolate, script, &parse_info, flag);
  }

  FinalizeUnoptimizedCompilation(isolate, script, flags, &compile_state,
                                 finalize_unoptimized_compilation_data_list);

  if (v8_flags.always_sparkplug) {
    CompileAllWithBaseline(isolate, finalize_unoptimized_compilation_data_list);
  }

  DCHECK(!isolate->has_pending_exception());
  DCHECK(is_compiled_scope->is_compiled());
  return true;
}
