  V8_INLINE Value Pop(int index, ValueType expected) {
    auto val = Pop();
    if (!VALIDATE(val.type == expected || val.type == kWasmVar ||
                  expected == kWasmVar)) {
      this->errorf(val.pc, "%s[%d] expected type %s, found %s of type %s",
                   SafeOpcodeNameAt(this->pc_), index,
                   ValueTypes::TypeName(expected), SafeOpcodeNameAt(val.pc),
                   ValueTypes::TypeName(val.type));
    }
    return val;
  }
