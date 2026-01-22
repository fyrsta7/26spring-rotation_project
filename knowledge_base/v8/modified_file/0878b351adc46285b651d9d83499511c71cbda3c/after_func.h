  OpIndex REDUCE(Store)(OpIndex base_idx, OptionalOpIndex index, OpIndex value,
                        StoreOp::Kind kind, MemoryRepresentation stored_rep,
                        WriteBarrierKind write_barrier, int32_t offset,
                        uint8_t element_scale,
                        bool maybe_initializing_or_transitioning,
                        IndirectPointerTag maybe_indirect_pointer_tag) {
    LABEL_BLOCK(no_change) {
      return Next::ReduceStore(base_idx, index, value, kind, stored_rep,
                               write_barrier, offset, element_scale,
                               maybe_initializing_or_transitioning,
                               maybe_indirect_pointer_tag);
    }
    if (ShouldSkipOptimizationStep()) goto no_change;

    if (stored_rep.SizeInBytes() <= 4) {
      value = TryRemoveWord32ToWord64Conversion(value);
    }
    index =
        ReduceMemoryIndex(index.value_or_invalid(), &offset, &element_scale);
    switch (stored_rep) {
      case MemoryRepresentation::Uint8():
      case MemoryRepresentation::Int8():
        value = ReduceWithTruncation(value, std::numeric_limits<uint8_t>::max(),
                                     WordRepresentation::Word32());
        break;
      case MemoryRepresentation::Uint16():
      case MemoryRepresentation::Int16():
        value =
            ReduceWithTruncation(value, std::numeric_limits<uint16_t>::max(),
                                 WordRepresentation::Word32());
        break;
      case MemoryRepresentation::Uint32():
      case MemoryRepresentation::Int32():
        value =
            ReduceWithTruncation(value, std::numeric_limits<uint32_t>::max(),
                                 WordRepresentation::Word32());
        break;
      default:
        break;
    }

    // If index is invalid and base is `left+right`, we use `left` as base and
    // `right` as index.
    if (!index.valid() && matcher.Is<Opmask::kWord64Add>(base_idx)) {
      DCHECK_EQ(element_scale, 0);
      const WordBinopOp& base = matcher.Cast<WordBinopOp>(base_idx);
      base_idx = base.left();
      index = base.right();
      // We go through the Store stack again, which might merge {index} into
      // {offset}, or just do other optimizations on this Store.
      __ Store(base_idx, index, value, kind, stored_rep, write_barrier, offset,
               element_scale, maybe_initializing_or_transitioning,
               maybe_indirect_pointer_tag);
      return OpIndex::Invalid();
    }

    return Next::ReduceStore(base_idx, index, value, kind, stored_rep,
                             write_barrier, offset, element_scale,
                             maybe_initializing_or_transitioning,
                             maybe_indirect_pointer_tag);
  }
