void HInferRepresentation::Analyze() {
  HPhase phase("Infer representations", graph_);

  // (1) Initialize bit vectors and count real uses. Each phi gets a
  // bit-vector of length <number of phis>.
  const ZoneList<HPhi*>* phi_list = graph_->phi_list();
  int phi_count = phi_list->length();
  ZoneList<BitVector*> connected_phis(phi_count);
  for (int i = 0; i < phi_count; ++i) {
    phi_list->at(i)->InitRealUses(i);
    BitVector* connected_set = new(zone()) BitVector(phi_count);
    connected_set->Add(i);
    connected_phis.Add(connected_set);
  }

  // (2) Do a fixed point iteration to find the set of connected phis.  A
  // phi is connected to another phi if its value is used either directly or
  // indirectly through a transitive closure of the def-use relation.
  bool change = true;
  while (change) {
    change = false;
    for (int i = 0; i < phi_count; ++i) {
      HPhi* phi = phi_list->at(i);
      for (HUseIterator it(phi->uses()); !it.Done(); it.Advance()) {
        HValue* use = it.value();
        if (use->IsPhi()) {
          int id = HPhi::cast(use)->phi_id();
          if (connected_phis[i]->UnionIsChanged(*connected_phis[id]))
            change = true;
        }
      }
    }
  }

  // (3) Sum up the non-phi use counts of all connected phis.  Don't include
  // the non-phi uses of the phi itself.
  for (int i = 0; i < phi_count; ++i) {
    HPhi* phi = phi_list->at(i);
    for (BitVector::Iterator it(connected_phis.at(i));
         !it.Done();
         it.Advance()) {
      int index = it.Current();
      if (index != i) {
        HPhi* it_use = phi_list->at(it.Current());
        phi->AddNonPhiUsesFrom(it_use);
      }
    }
  }

  // (4) Compute phis that definitely can't be converted to integer
  // without deoptimization and mark them to avoid unnecessary deoptimization.
  change = true;
  while (change) {
    change = false;
    for (int i = 0; i < phi_count; ++i) {
      HPhi* phi = phi_list->at(i);
      for (int j = 0; j < phi->OperandCount(); ++j) {
        if (phi->IsConvertibleToInteger() &&
            !phi->OperandAt(j)->IsConvertibleToInteger()) {
          phi->set_is_convertible_to_integer(false);
          change = true;
          break;
        }
      }
    }
  }


  for (int i = 0; i < graph_->blocks()->length(); ++i) {
    HBasicBlock* block = graph_->blocks()->at(i);
    const ZoneList<HPhi*>* phis = block->phis();
    for (int j = 0; j < phis->length(); ++j) {
      AddToWorklist(phis->at(j));
    }

    HInstruction* current = block->first();
    while (current != NULL) {
      AddToWorklist(current);
      current = current->next();
    }
  }

  while (!worklist_.is_empty()) {
    HValue* current = worklist_.RemoveLast();
    in_worklist_.Remove(current->id());
    InferBasedOnInputs(current);
    InferBasedOnUses(current);
  }
}
