void HOptimizedGraphBuilder::HandlePolymorphicLoadNamedField(
    int position,
    BailoutId ast_id,
    BailoutId return_id,
    HValue* object,
    SmallMapList* types,
    Handle<String> name) {
  // Something did not match; must use a polymorphic load.
  int count = 0;
  HBasicBlock* join = NULL;
  for (int i = 0; i < types->length() && count < kMaxLoadPolymorphism; ++i) {
    PropertyAccessInfo info(isolate(), types->at(i), name);
    if (info.CanLoadMonomorphic()) {
      if (count == 0) {
        BuildCheckHeapObject(object);
        join = graph()->CreateBasicBlock();
      }
      ++count;
      HBasicBlock* if_true = graph()->CreateBasicBlock();
      HBasicBlock* if_false = graph()->CreateBasicBlock();
      HCompareMap* compare = New<HCompareMap>(
          object, info.map(),  if_true, if_false);
      current_block()->Finish(compare);

      set_current_block(if_true);

      HInstruction* load = BuildLoadMonomorphic(
          &info, object, compare, ast_id, return_id, false);
      if (load == NULL) {
        if (HasStackOverflow()) return;
      } else {
        if (!load->IsLinked()) {
          load->set_position(position);
          AddInstruction(load);
        }
        if (!ast_context()->IsEffect()) Push(load);
      }

      if (current_block() != NULL) current_block()->Goto(join);
      set_current_block(if_false);
    }
  }

  // Finish up.  Unconditionally deoptimize if we've handled all the maps we
  // know about and do not want to handle ones we've never seen.  Otherwise
  // use a generic IC.
  if (count == types->length() && FLAG_deoptimize_uncommon_cases) {
    FinishExitWithHardDeoptimization("Unknown map in polymorphic load", join);
  } else {
    HValue* context = environment()->context();
    HInstruction* load = new(zone()) HLoadNamedGeneric(context, object, name);
    load->set_position(position);
    AddInstruction(load);
    if (!ast_context()->IsEffect()) Push(load);

    if (join != NULL) {
      current_block()->Goto(join);
    } else {
      Add<HSimulate>(ast_id, REMOVABLE_SIMULATE);
      if (!ast_context()->IsEffect()) ast_context()->ReturnValue(Pop());
      return;
    }
  }

  ASSERT(join != NULL);
  join->SetJoinId(ast_id);
  set_current_block(join);
  if (!ast_context()->IsEffect()) ast_context()->ReturnValue(Pop());
}
