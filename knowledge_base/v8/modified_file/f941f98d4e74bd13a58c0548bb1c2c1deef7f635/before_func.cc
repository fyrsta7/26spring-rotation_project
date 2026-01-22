void LiftoffAssembler::PrepareCall(const ValueKindSig* sig,
                                   compiler::CallDescriptor* call_descriptor,
                                   Register* target,
                                   Register* target_instance) {
  uint32_t num_params = static_cast<uint32_t>(sig->parameter_count());
  // Input 0 is the call target.
  constexpr size_t kInputShift = 1;

  // Spill all cache slots which are not being used as parameters.
  cache_state_.ClearAllCacheRegisters();
  for (VarState* it = cache_state_.stack_state.end() - 1 - num_params;
       it >= cache_state_.stack_state.begin() &&
       !cache_state_.used_registers.is_empty();
       --it) {
    if (!it->is_reg()) continue;
    Spill(it->offset(), it->reg(), it->kind());
    cache_state_.dec_used(it->reg());
    it->MakeStack();
  }

  LiftoffStackSlots stack_slots(this);
  StackTransferRecipe stack_transfers(this);
  LiftoffRegList param_regs;

  // Move the target instance (if supplied) into the correct instance register.
  compiler::LinkageLocation instance_loc =
      call_descriptor->GetInputLocation(kInputShift);
  DCHECK(instance_loc.IsRegister() && !instance_loc.IsAnyRegister());
  Register instance_reg = Register::from_code(instance_loc.AsRegister());
  param_regs.set(instance_reg);
  if (target_instance && *target_instance != instance_reg) {
    stack_transfers.MoveRegister(LiftoffRegister(instance_reg),
                                 LiftoffRegister(*target_instance),
                                 kIntPtrKind);
  }

  int param_slots = static_cast<int>(call_descriptor->ParameterSlotCount());
  if (num_params) {
    uint32_t param_base = cache_state_.stack_height() - num_params;
    PrepareStackTransfers(sig, call_descriptor,
                          &cache_state_.stack_state[param_base], &stack_slots,
                          &stack_transfers, &param_regs);
  }

  // If the target register overlaps with a parameter register, then move the
  // target to another free register, or spill to the stack.
  if (target && param_regs.has(LiftoffRegister(*target))) {
    // Try to find another free register.
    LiftoffRegList free_regs = kGpCacheRegList.MaskOut(param_regs);
    if (!free_regs.is_empty()) {
      LiftoffRegister new_target = free_regs.GetFirstRegSet();
      stack_transfers.MoveRegister(new_target, LiftoffRegister(*target),
                                   kIntPtrKind);
      *target = new_target.gp();
    } else {
      stack_slots.Add(VarState(kIntPtrKind, LiftoffRegister(*target), 0),
                      param_slots);
      param_slots++;
      *target = no_reg;
    }
  }

  if (param_slots > 0) {
    stack_slots.Construct(param_slots);
  }
  // Execute the stack transfers before filling the instance register.
  stack_transfers.Execute();
  // Pop parameters from the value stack.
  cache_state_.stack_state.pop_back(num_params);

  // Reset register use counters.
  cache_state_.reset_used_registers();

  // Reload the instance from the stack.
  if (!target_instance) {
    LoadInstanceFromFrame(instance_reg);
  }
}
