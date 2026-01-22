  Trace greedy_match_trace;
  if (not_at_start()) greedy_match_trace.set_at_start(Trace::FALSE_VALUE);
  greedy_match_trace.set_backtrack(&greedy_match_failed);
  Label loop_label;
  macro_assembler->Bind(&loop_label);
  greedy_match_trace.set_stop_node(this);
  greedy_match_trace.set_loop_label(&loop_label);
  alternatives_->at(0).node()->Emit(compiler, &greedy_match_trace);
  macro_assembler->Bind(&greedy_match_failed);

  Label second_choice;  // For use in greedy matches.
  macro_assembler->Bind(&second_choice);

  Trace* new_trace = greedy_loop_state->counter_backtrack_trace();

  EmitChoices(compiler, alt_gens, 1, new_trace, preload);

  macro_assembler->Bind(greedy_loop_state->label());
  // If we have unwound to the bottom then backtrack.
  macro_assembler->CheckGreedyLoop(trace->backtrack());
  // Otherwise try the second priority at an earlier position.
  macro_assembler->AdvanceCurrentPosition(-text_length);
  macro_assembler->GoTo(&second_choice);
  return new_trace;
}

int ChoiceNode::EmitOptimizedUnanchoredSearch(RegExpCompiler* compiler,
                                              Trace* trace) {
  int eats_at_least = PreloadState::kEatsAtLeastNotYetInitialized;
  if (alternatives_->length() != 2) return eats_at_least;

  GuardedAlternative alt1 = alternatives_->at(1);
  if (alt1.guards() != nullptr && alt1.guards()->length() != 0) {
    return eats_at_least;
  }
  RegExpNode* eats_anything_node = alt1.node();
  if (eats_anything_node->GetSuccessorOfOmnivorousTextNode(compiler) != this) {
    return eats_at_least;
  }

  // Really we should be creating a new trace when we execute this function,
  // but there is no need, because the code it generates cannot backtrack, and
  // we always arrive here with a trivial trace (since it's the entry to a
  // loop.  That also implies that there are no preloaded characters, which is
  // good, because it means we won't be violating any assumptions by
  // overwriting those characters with new load instructions.
  DCHECK(trace->is_trivial());

  RegExpMacroAssembler* macro_assembler = compiler->macro_assembler();
  Isolate* isolate = macro_assembler->isolate();
  // At this point we know that we are at a non-greedy loop that will eat
  // any character one at a time.  Any non-anchored regexp has such a
  // loop prepended to it in order to find where it starts.  We look for
  // a pattern of the form ...abc... where we can look 6 characters ahead
  // and step forwards 3 if the character is not one of abc.  Abc need
  // not be atoms, they can be any reasonably limited character class or
  // small alternation.
  BoyerMooreLookahead* bm = bm_info(false);
  if (bm == nullptr) {
    eats_at_least = std::min(kMaxLookaheadForBoyerMoore, EatsAtLeast(false));
    if (eats_at_least >= 1) {
      bm = zone()->New<BoyerMooreLookahead>(eats_at_least, compiler, zone());
      GuardedAlternative alt0 = alternatives_->at(0);
      alt0.node()->FillInBMInfo(isolate, 0, kRecursionBudget, bm, false);
    }
  }
  if (bm != nullptr) {
    bm->EmitSkipInstructions(macro_assembler);
  }
  return eats_at_least;
}

void ChoiceNode::EmitChoices(RegExpCompiler* compiler,
                             AlternativeGenerationList* alt_gens,
                             int first_choice, Trace* trace,
                             PreloadState* preload) {
  RegExpMacroAssembler* macro_assembler = compiler->macro_assembler();
  SetUpPreLoad(compiler, trace, preload);

  // For now we just call all choices one after the other.  The idea ultimately
  // is to use the Dispatch table to try only the relevant ones.
  int choice_count = alternatives_->length();
