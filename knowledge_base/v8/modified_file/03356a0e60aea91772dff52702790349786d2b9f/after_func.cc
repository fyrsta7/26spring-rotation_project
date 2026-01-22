static Object* Runtime_NumberToRadixString(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  // Fast case where the result is a one character string.
  if (args[0]->IsSmi() && args[1]->IsSmi()) {
    int value = Smi::cast(args[0])->value();
    int radix = Smi::cast(args[1])->value();
    if (value >= 0 && value < radix) {
      RUNTIME_ASSERT(radix <= 36);
      // Character array used for conversion.
      static const char kCharTable[] = "0123456789abcdefghijklmnopqrstuvwxyz";
      return Heap::LookupSingleCharacterStringFromCode(kCharTable[value]);
    }
  }

  // Slow case.
  CONVERT_DOUBLE_CHECKED(value, args[0]);
  if (isnan(value)) {
    return Heap::AllocateStringFromAscii(CStrVector("NaN"));
  }
  if (isinf(value)) {
    if (value < 0) {
      return Heap::AllocateStringFromAscii(CStrVector("-Infinity"));
    }
    return Heap::AllocateStringFromAscii(CStrVector("Infinity"));
  }
  CONVERT_DOUBLE_CHECKED(radix_number, args[1]);
  int radix = FastD2I(radix_number);
  RUNTIME_ASSERT(2 <= radix && radix <= 36);
  char* str = DoubleToRadixCString(value, radix);
  Object* result = Heap::AllocateStringFromAscii(CStrVector(str));
  DeleteArray(str);
  return result;
}
