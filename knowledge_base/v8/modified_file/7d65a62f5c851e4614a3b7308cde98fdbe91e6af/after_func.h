  inline void ExtractBitRange(Register dst, Register src, int rangeStart,
                              int rangeEnd, RCBit rc = LeaveRC) {
    DCHECK(rangeStart >= rangeEnd && rangeStart < kBitsPerPointer);
    int rotate = (rangeEnd == 0) ? 0 : kBitsPerPointer - rangeEnd;
    int width = rangeStart - rangeEnd + 1;
    if (rc == SetRC && rangeEnd == 0 && width <= 16) {
      andi(dst, src, Operand((1 << width) - 1));
    } else {
#if V8_TARGET_ARCH_PPC64
      rldicl(dst, src, rotate, kBitsPerPointer - width, rc);
#else
      rlwinm(dst, src, rotate, kBitsPerPointer - width, kBitsPerPointer - 1,
             rc);
#endif
    }
  }
