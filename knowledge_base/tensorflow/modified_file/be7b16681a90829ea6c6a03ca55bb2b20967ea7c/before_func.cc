    }
    return OkStatus();
  }

  Status HandleCustomCall(HloInstruction* hlo) override {
    if (opts_.verify_custom_call_nested_computation_thread_name) {
      // Allow kCustomCall to contain computations on separate thread.
      return CheckCallableInstructionThreadName(
          hlo, /*skip_nested_async_op_check=*/true);
    }
    return OkStatus();
  }

  Status Preprocess(HloInstruction* instruction) override {
    auto previous = instructions_by_name_.find(instruction->name());
    TF_RET_CHECK(previous == instructions_by_name_.end())
        << "HLO has name that is not unique within module:\n"
        << instruction->ToString()
        << " in computation: " << instruction->parent()->name()
        << "\nPrevious HLO with same name:\n"
        << previous->second->ToString()
        << " in computation: " << previous->second->parent()->name();
    instructions_by_name_[instruction->name()] = instruction;

