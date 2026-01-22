HGraph* HGraphBuilder::CreateGraph() {
  graph_ = new(zone()) HGraph(info());
  if (FLAG_hydrogen_stats) HStatistics::Instance()->Initialize(info());

  {
    HPhase phase("Block building");
    current_block_ = graph()->entry_block();

    Scope* scope = info()->scope();
    if (scope->HasIllegalRedeclaration()) {
      Bailout("function with illegal redeclaration");
      return NULL;
    }
    SetupScope(scope);
    VisitDeclarations(scope->declarations());
    HValue* context = environment()->LookupContext();
    AddInstruction(
        new(zone()) HStackCheck(context, HStackCheck::kFunctionEntry));

    // Add an edge to the body entry.  This is warty: the graph's start
    // environment will be used by the Lithium translation as the initial
    // environment on graph entry, but it has now been mutated by the
    // Hydrogen translation of the instructions in the start block.  This
    // environment uses values which have not been defined yet.  These
    // Hydrogen instructions will then be replayed by the Lithium
    // translation, so they cannot have an environment effect.  The edge to
    // the body's entry block (along with some special logic for the start
    // block in HInstruction::InsertAfter) seals the start block from
    // getting unwanted instructions inserted.
    //
    // TODO(kmillikin): Fix this.  Stop mutating the initial environment.
    // Make the Hydrogen instructions in the initial block into Hydrogen
    // values (but not instructions), present in the initial environment and
    // not replayed by the Lithium translation.
    HEnvironment* initial_env = environment()->CopyWithoutHistory();
    HBasicBlock* body_entry = CreateBasicBlock(initial_env);
    current_block()->Goto(body_entry);
    body_entry->SetJoinId(AstNode::kFunctionEntryId);
    set_current_block(body_entry);
    VisitStatements(info()->function()->body());
    if (HasStackOverflow()) return NULL;

    if (current_block() != NULL) {
      HReturn* instr = new(zone()) HReturn(graph()->GetConstantUndefined());
      current_block()->FinishExit(instr);
      set_current_block(NULL);
    }
  }

  graph()->OrderBlocks();
  graph()->AssignDominators();
  graph()->PropagateDeoptimizingMark();
  graph()->EliminateRedundantPhis();
  if (!graph()->CheckPhis()) {
    Bailout("Unsupported phi use of arguments object");
    return NULL;
  }
  if (FLAG_eliminate_dead_phis) graph()->EliminateUnreachablePhis();
  if (!graph()->CollectPhis()) {
    Bailout("Unsupported phi use of uninitialized constant");
    return NULL;
  }

  HInferRepresentation rep(graph());
  rep.Analyze();

  graph()->MarkDeoptimizeOnUndefined();
  graph()->InsertRepresentationChanges();

  graph()->InitializeInferredTypes();
  graph()->Canonicalize();

  // Perform common subexpression elimination and loop-invariant code motion.
  if (FLAG_use_gvn) {
    HPhase phase("Global value numbering", graph());
    HGlobalValueNumberer gvn(graph(), info());
    gvn.Analyze();
  }

  if (FLAG_use_range) {
    HRangeAnalysis rangeAnalysis(graph());
    rangeAnalysis.Analyze();
  }
  graph()->ComputeMinusZeroChecks();

  // Eliminate redundant stack checks on backwards branches.
  HStackCheckEliminator sce(graph());
  sce.Process();

  // Replace the results of check instructions with the original value, if the
  // result is used. This is safe now, since we don't do code motion after this
  // point. It enables better register allocation since the value produced by
  // check instructions is really a copy of the original value.
  graph()->ReplaceCheckedValues();

  return graph();
}
