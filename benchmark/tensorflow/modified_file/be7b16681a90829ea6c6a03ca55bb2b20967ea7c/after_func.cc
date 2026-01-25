  Status Preprocess(HloInstruction* instruction) override {
    auto [it, inserted] =
        instructions_by_name_.insert({instruction->name(), instruction});
    TF_RET_CHECK(inserted) << "HLO has name that is not unique within module:\n"
                           << instruction->ToString() << " in computation: "
                           << instruction->parent()->name()
                           << "\nPrevious HLO with same name:\n"
                           << it->second->ToString() << " in computation: "
                           << it->second->parent()->name();

    if (instruction->has_sharding()) {
      Status status =
          instruction->sharding().Validate(instruction->shape(), num_devices_);
      if (!status.ok()) {
        return Status(status.code(),
                      absl::StrCat("Invalid sharding for instruction: ",
                                   instruction->ToString(), ": ",
                                   status.error_message()));
      }
    }

    return OkStatus();
  }