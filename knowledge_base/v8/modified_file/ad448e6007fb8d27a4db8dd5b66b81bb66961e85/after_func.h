  template <typename IntType, typename ValidationTag, TraceFlag trace,
            size_t size_in_bits = 8 * sizeof(IntType)>
  V8_INLINE IntType read_leb(const byte* pc, uint32_t* length,
                             Name<ValidationTag> name = "varint") {
    static_assert(size_in_bits <= 8 * sizeof(IntType),
                  "leb does not fit in type");
    TRACE_IF(trace, "  +%u  %-20s: ", pc_offset(),
             implicit_cast<const char*>(name));
    // Fast path for single-byte integers.
    if (V8_LIKELY((!ValidationTag::validate || pc < end_) && !(*pc & 0x80))) {
      TRACE_IF(trace, "%02x ", *pc);
      *length = 1;
      IntType result = *pc;
      if (std::is_signed<IntType>::value) {
        // Perform sign extension.
        constexpr int sign_ext_shift = int{8 * sizeof(IntType)} - 7;
        result = (result << sign_ext_shift) >> sign_ext_shift;
        TRACE_IF(trace, "= %" PRIi64 "\n", static_cast<int64_t>(result));
      } else {
        TRACE_IF(trace, "= %" PRIu64 "\n", static_cast<uint64_t>(result));
      }
      return result;
    }
    IntType result;
    read_leb_slowpath<IntType, ValidationTag, trace, size_in_bits>(
        pc, length, name, &result);
    return result;
  }
