MaybeHandle<JSFunction> Compiler::GetFunctionFromEval(
    Handle<String> source, Handle<SharedFunctionInfo> outer_info,
    Handle<Context> context, LanguageMode language_mode,
    ParseRestriction restriction, int eval_scope_position, int eval_position,
    int line_offset, int column_offset, Handle<Object> script_name,
    ScriptOriginOptions options) {
  Isolate* isolate = source->GetIsolate();
  int source_length = source->length();
  isolate->counters()->total_eval_size()->Increment(source_length);
  isolate->counters()->total_compile_size()->Increment(source_length);

  CompilationCache* compilation_cache = isolate->compilation_cache();
  MaybeHandle<SharedFunctionInfo> maybe_shared_info =
      compilation_cache->LookupEval(source, outer_info, context, language_mode,
                                    eval_scope_position);
  Handle<SharedFunctionInfo> shared_info;

  Handle<Script> script;
  if (!maybe_shared_info.ToHandle(&shared_info)) {
    script = isolate->factory()->NewScript(source);
    if (!script_name.is_null()) {
      script->set_name(*script_name);
      script->set_line_offset(line_offset);
      script->set_column_offset(column_offset);
    }
    script->set_origin_options(options);
    script->set_compilation_type(Script::COMPILATION_TYPE_EVAL);
    Script::SetEvalOrigin(script, outer_info, eval_position);

    Zone zone(isolate->allocator());
    ParseInfo parse_info(&zone, script);
    CompilationInfo info(&parse_info, Handle<JSFunction>::null());
    parse_info.set_eval();
    if (context->IsNativeContext()) parse_info.set_global();
    parse_info.set_language_mode(language_mode);
    parse_info.set_parse_restriction(restriction);
    parse_info.set_context(context);

    shared_info = CompileToplevel(&info);

    if (shared_info.is_null()) {
      return MaybeHandle<JSFunction>();
    } else {
      // If caller is strict mode, the result must be in strict mode as well.
      DCHECK(is_sloppy(language_mode) ||
             is_strict(shared_info->language_mode()));
      compilation_cache->PutEval(source, outer_info, context, shared_info,
                                 eval_scope_position);
    }
  }

  Handle<JSFunction> result =
      isolate->factory()->NewFunctionFromSharedFunctionInfo(
          shared_info, context, NOT_TENURED);

  // OnAfterCompile has to be called after we create the JSFunction, which we
  // may require to recompile the eval for debugging, if we find a function
  // that contains break points in the eval script.
  isolate->debug()->OnAfterCompile(script);

  return result;
}
