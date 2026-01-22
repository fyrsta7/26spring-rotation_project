bool HGraphBuilder::TryInline(CallKind call_kind,
                              Handle<JSFunction> target,
                              ZoneList<Expression*>* arguments,
                              HValue* receiver,
                              int ast_id,
                              int return_id,
                              ReturnHandlingFlag return_handling) {
  if (!FLAG_use_inlining) return false;

  // Precondition: call is monomorphic and we have found a target with the
  // appropriate arity.
  Handle<JSFunction> caller = info()->closure();
  Handle<SharedFunctionInfo> target_shared(target->shared());

  // Do a quick check on source code length to avoid parsing large
  // inlining candidates.
  if ((FLAG_limit_inlining && target_shared->SourceSize() > kMaxSourceSize)
      || target_shared->SourceSize() > kUnlimitedMaxSourceSize) {
    TraceInline(target, caller, "target text too big");
    return false;
  }

  // Target must be inlineable.
  if (!target->IsInlineable()) {
    TraceInline(target, caller, "target not inlineable");
    return false;
  }
  if (target_shared->dont_inline() || target_shared->dont_optimize()) {
    TraceInline(target, caller, "target contains unsupported syntax [early]");
    return false;
  }

  int nodes_added = target_shared->ast_node_count();
  if ((FLAG_limit_inlining && nodes_added > kMaxInlinedSize) ||
      nodes_added > kUnlimitedMaxInlinedSize) {
    TraceInline(target, caller, "target AST is too large [early]");
    return false;
  }

#if !defined(V8_TARGET_ARCH_IA32)
  // Target must be able to use caller's context.
  CompilationInfo* outer_info = info();
  if (target->context() != outer_info->closure()->context() ||
      outer_info->scope()->contains_with() ||
      outer_info->scope()->num_heap_slots() > 0) {
    TraceInline(target, caller, "target requires context change");
    return false;
  }
#endif


  // Don't inline deeper than kMaxInliningLevels calls.
  HEnvironment* env = environment();
  int current_level = 1;
  while (env->outer() != NULL) {
    if (current_level == Compiler::kMaxInliningLevels) {
      TraceInline(target, caller, "inline depth limit reached");
      return false;
    }
    if (env->outer()->frame_type() == JS_FUNCTION) {
      current_level++;
    }
    env = env->outer();
  }

  // Don't inline recursive functions.
  for (FunctionState* state = function_state();
       state != NULL;
       state = state->outer()) {
    if (state->compilation_info()->closure()->shared() == *target_shared) {
      TraceInline(target, caller, "target is recursive");
      return false;
    }
  }

  // We don't want to add more than a certain number of nodes from inlining.
  if ((FLAG_limit_inlining && inlined_count_ > kMaxInlinedNodes) ||
      inlined_count_ > kUnlimitedMaxInlinedNodes) {
    TraceInline(target, caller, "cumulative AST node limit reached");
    return false;
  }

  // Parse and allocate variables.
  CompilationInfo target_info(target);
  if (!ParserApi::Parse(&target_info, kNoParsingFlags) ||
      !Scope::Analyze(&target_info)) {
    if (target_info.isolate()->has_pending_exception()) {
      // Parse or scope error, never optimize this function.
      SetStackOverflow();
      target_shared->DisableOptimization();
    }
    TraceInline(target, caller, "parse failure");
    return false;
  }

  if (target_info.scope()->num_heap_slots() > 0) {
    TraceInline(target, caller, "target has context-allocated variables");
    return false;
  }
  FunctionLiteral* function = target_info.function();

  // The following conditions must be checked again after re-parsing, because
  // earlier the information might not have been complete due to lazy parsing.
  nodes_added = function->ast_node_count();
  if ((FLAG_limit_inlining && nodes_added > kMaxInlinedSize) ||
      nodes_added > kUnlimitedMaxInlinedSize) {
    TraceInline(target, caller, "target AST is too large [late]");
    return false;
  }
  AstProperties::Flags* flags(function->flags());
  if (flags->Contains(kDontInline) || flags->Contains(kDontOptimize)) {
    TraceInline(target, caller, "target contains unsupported syntax [late]");
    return false;
  }

  // Don't inline functions that uses the arguments object.
  if (function->scope()->arguments() != NULL) {
    TraceInline(target, caller, "target requires special argument handling");
    return false;
  }

  // All declarations must be inlineable.
  ZoneList<Declaration*>* decls = target_info.scope()->declarations();
  int decl_count = decls->length();
  for (int i = 0; i < decl_count; ++i) {
    if (!decls->at(i)->IsInlineable()) {
      TraceInline(target, caller, "target has non-trivial declaration");
      return false;
    }
  }

  // Generate the deoptimization data for the unoptimized version of
  // the target function if we don't already have it.
  if (!target_shared->has_deoptimization_support()) {
    // Note that we compile here using the same AST that we will use for
    // generating the optimized inline code.
    target_info.EnableDeoptimizationSupport();
    if (!FullCodeGenerator::MakeCode(&target_info)) {
      TraceInline(target, caller, "could not generate deoptimization info");
      return false;
    }
    if (target_shared->scope_info() == ScopeInfo::Empty()) {
      // The scope info might not have been set if a lazily compiled
      // function is inlined before being called for the first time.
      Handle<ScopeInfo> target_scope_info =
          ScopeInfo::Create(target_info.scope());
      target_shared->set_scope_info(*target_scope_info);
    }
    target_shared->EnableDeoptimizationSupport(*target_info.code());
    Compiler::RecordFunctionCompilation(Logger::FUNCTION_TAG,
                                        &target_info,
                                        target_shared);
  }

  // ----------------------------------------------------------------
  // After this point, we've made a decision to inline this function (so
  // TryInline should always return true).

  // Save the pending call context and type feedback oracle. Set up new ones
  // for the inlined function.
  ASSERT(target_shared->has_deoptimization_support());
  TypeFeedbackOracle target_oracle(
      Handle<Code>(target_shared->code()),
      Handle<Context>(target->context()->global_context()),
      isolate());
  // The function state is new-allocated because we need to delete it
  // in two different places.
  FunctionState* target_state = new FunctionState(
      this, &target_info, &target_oracle, return_handling);

  HConstant* undefined = graph()->GetConstantUndefined();
  HEnvironment* inner_env =
      environment()->CopyForInlining(target,
                                     arguments->length(),
                                     function,
                                     undefined,
                                     call_kind,
                                     function_state()->is_construct());
#ifdef V8_TARGET_ARCH_IA32
  // IA32 only, overwrite the caller's context in the deoptimization
  // environment with the correct one.
  //
  // TODO(kmillikin): implement the same inlining on other platforms so we
  // can remove the unsightly ifdefs in this function.
  HConstant* context = new HConstant(Handle<Context>(target->context()),
                                     Representation::Tagged());
  AddInstruction(context);
  inner_env->BindContext(context);
#endif
  HBasicBlock* body_entry = CreateBasicBlock(inner_env);
  current_block()->Goto(body_entry);
  body_entry->SetJoinId(return_id);
  set_current_block(body_entry);
  AddInstruction(new(zone()) HEnterInlined(target,
                                           arguments->length(),
                                           function,
                                           call_kind,
                                           function_state()->is_construct()));
  VisitDeclarations(target_info.scope()->declarations());
  VisitStatements(function->body());
  if (HasStackOverflow()) {
    // Bail out if the inline function did, as we cannot residualize a call
    // instead.
    TraceInline(target, caller, "inline graph construction failed");
    target_shared->DisableOptimization();
    inline_bailout_ = true;
    delete target_state;
    return true;
  }

  // Update inlined nodes count.
  inlined_count_ += nodes_added;

  TraceInline(target, caller, NULL);

  if (current_block() != NULL) {
    // Add default return value (i.e. undefined for normals calls or the newly
    // allocated receiver for construct calls) if control can fall off the
    // body.  In a test context, undefined is false and any JSObject is true.
    if (call_context()->IsValue()) {
      ASSERT(function_return() != NULL);
      HValue* return_value = function_state()->is_construct()
          ? receiver
          : undefined;
      current_block()->AddLeaveInlined(return_value,
                                       function_return(),
                                       function_state()->drop_extra());
    } else if (call_context()->IsEffect()) {
      ASSERT(function_return() != NULL);
      current_block()->Goto(function_return(), function_state()->drop_extra());
    } else {
      ASSERT(call_context()->IsTest());
      ASSERT(inlined_test_context() != NULL);
      HBasicBlock* target = function_state()->is_construct()
          ? inlined_test_context()->if_true()
          : inlined_test_context()->if_false();
      current_block()->Goto(target, function_state()->drop_extra());
    }
  }

  // Fix up the function exits.
  if (inlined_test_context() != NULL) {
    HBasicBlock* if_true = inlined_test_context()->if_true();
    HBasicBlock* if_false = inlined_test_context()->if_false();

    // Pop the return test context from the expression context stack.
    ASSERT(ast_context() == inlined_test_context());
    ClearInlinedTestContext();
    delete target_state;

    // Forward to the real test context.
    if (if_true->HasPredecessor()) {
      if_true->SetJoinId(ast_id);
      HBasicBlock* true_target = TestContext::cast(ast_context())->if_true();
      if_true->Goto(true_target, function_state()->drop_extra());
    }
    if (if_false->HasPredecessor()) {
      if_false->SetJoinId(ast_id);
      HBasicBlock* false_target = TestContext::cast(ast_context())->if_false();
      if_false->Goto(false_target, function_state()->drop_extra());
    }
    set_current_block(NULL);
    return true;

  } else if (function_return()->HasPredecessor()) {
    function_return()->SetJoinId(ast_id);
    set_current_block(function_return());
  } else {
    set_current_block(NULL);
  }
  delete target_state;
  return true;
}
