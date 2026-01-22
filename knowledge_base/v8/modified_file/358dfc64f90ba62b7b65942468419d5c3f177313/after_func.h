  template <StackElementsCountMode strict_count, bool push_branch_values,
            MergeType merge_type>
  V8_INLINE bool TypeCheckStackAgainstMerge(Merge<Value>* merge) {
    uint32_t arity = merge->arity;
    uint32_t actual = stack_.size() - control_.back().stack_depth;
    // Handle trivial cases first. Arity 0 is the most common case.
    if (arity == 0 && (!strict_count || actual == 0)) return true;
    // Arity 1 is still common enough that we handle it separately (only doing
    // the most basic subtype check).
    if (arity == 1 && (strict_count ? actual == arity : actual >= arity)) {
      if (stack_.back().type == merge->vals.first.type) return true;
    }
    return TypeCheckStackAgainstMerge_Slow<strict_count, push_branch_values,
                                           merge_type>(merge);
  }
