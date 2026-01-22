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
    auto [it, inserted] =
        instructions_by_name_.insert({instruction->name(), instruction});
    TF_RET_CHECK(inserted) << "HLO has name that is not unique within module:\n"
                           << instruction->ToString() << " in computation: "
                           << instruction->parent()->name()
                           << "\nPrevious HLO with same name:\n"
                           << it->second->ToString() << " in computation: "
                           << it->second->parent()->name();

