  void SimplifyLoadStore(OpIndex& base, OptionalOpIndex& index,
                         LoadOp::Kind& kind, int32_t& offset,
                         uint8_t& element_size_log2) {
    if (!lowering_enabled_) return;

    if (element_size_log2 > kMaxElementSizeLog2) {
      DCHECK(index.valid());
      index = __ WordPtrShiftLeft(index.value(), element_size_log2);
      element_size_log2 = 0;
    }

    if (kNeedsUntaggedBase) {
      if (kind.tagged_base) {
        kind.tagged_base = false;
        DCHECK_LE(std::numeric_limits<int32_t>::min() + kHeapObjectTag, offset);
        offset -= kHeapObjectTag;
        base = __ BitcastHeapObjectToWordPtr(base);
      }
    }

    // TODO(nicohartmann@): Remove the case for atomics once crrev.com/c/5237267
    // is ported to x64.
    if (!CanEncodeOffset(offset, kind.tagged_base) ||
        (kind.is_atomic && !CanEncodeAtomic(index, offset))) {
      // If an index is present, the element_size_log2 is changed to zero.
      // So any load follows the form *(base + offset). To simplify
      // instruction selection, both static and dynamic offsets are stored in
      // the index input.
      // As tagged loads result in modifying the offset by -1, those loads are
      // converted into raw loads (above).
      if (!index.has_value() || matcher_.MatchIntegralZero(index.value())) {
        index = __ IntPtrConstant(offset);
        element_size_log2 = 0;
        offset = 0;
      }
      if (element_size_log2 != 0) {
        index = __ WordPtrShiftLeft(index.value(), element_size_log2);
        element_size_log2 = 0;
      }
      if (offset != 0) {
        index = __ WordPtrAdd(index.value(), offset);
        offset = 0;
      }
      DCHECK_EQ(offset, 0);
      DCHECK_EQ(element_size_log2, 0);
    }
  }
