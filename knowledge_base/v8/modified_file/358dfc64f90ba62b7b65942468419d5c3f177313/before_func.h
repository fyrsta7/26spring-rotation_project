  template <StackElementsCountMode strict_count, bool push_branch_values,
            MergeType merge_type>
  bool TypeCheckStackAgainstMerge(Merge<Value>* merge) {
    constexpr const char* merge_description =
        merge_type == kBranchMerge     ? "branch"
        : merge_type == kReturnMerge   ? "return"
        : merge_type == kInitExprMerge ? "constant expression"
                                       : "fallthru";
    uint32_t arity = merge->arity;
    uint32_t actual = stack_.size() - control_.back().stack_depth;
    // Here we have to check for !unreachable(), because we need to typecheck as
    // if the current code is reachable even if it is spec-only reachable.
    if (V8_LIKELY(decoding_mode == kConstantExpression ||
                  !control_.back().unreachable())) {
      if (V8_UNLIKELY(strict_count ? actual != arity : actual < arity)) {
        this->DecodeError("expected %u elements on the stack for %s, found %u",
                          arity, merge_description, actual);
        return false;
      }
      // Typecheck the topmost {merge->arity} values on the stack.
      Value* stack_values = stack_.end() - arity;
      for (uint32_t i = 0; i < arity; ++i) {
        Value& val = stack_values[i];
        Value& old = (*merge)[i];
        if (!IsSubtypeOf(val.type, old.type, this->module_)) {
          this->DecodeError("type error in %s[%u] (expected %s, got %s)",
                            merge_description, i, old.type.name().c_str(),
                            val.type.name().c_str());
          return false;
        }
      }
      return true;
    }
    // Unreachable code validation starts here.
    if (V8_UNLIKELY(strict_count && actual > arity)) {
      this->DecodeError("expected %u elements on the stack for %s, found %u",
                        arity, merge_description, actual);
      return false;
    }
    // TODO(manoskouk): Use similar code as above if we keep unreachable checks.
    for (int i = arity - 1, depth = 0; i >= 0; --i, ++depth) {
      Peek(depth, i, (*merge)[i].type);
    }
    if constexpr (push_branch_values) {
      uint32_t inserted_value_count =
          static_cast<uint32_t>(EnsureStackArguments(arity));
      if (inserted_value_count > 0) {
        // stack_.EnsureMoreCapacity() may have inserted unreachable values into
        // the bottom of the stack. If so, mark them with the correct type. If
        // drop values were also inserted, disregard them, as they will be
        // dropped anyway.
        Value* stack_base = stack_value(arity);
        for (uint32_t i = 0; i < std::min(arity, inserted_value_count); i++) {
          if (stack_base[i].type == kWasmBottom) {
            stack_base[i].type = (*merge)[i].type;
          }
        }
      }
    }
    return VALIDATE(this->ok());
  }
