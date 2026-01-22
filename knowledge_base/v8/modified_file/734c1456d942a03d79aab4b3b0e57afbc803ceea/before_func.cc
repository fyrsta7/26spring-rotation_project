icu::UnicodeString Intl::ToICUUnicodeString(Isolate* isolate,
                                            Handle<String> string) {
  DCHECK(string->IsFlat());
  DisallowHeapAllocation no_gc;
  std::unique_ptr<uc16[]> sap;
  return icu::UnicodeString(
      GetUCharBufferFromFlat(string->GetFlatContent(no_gc), &sap,
                             string->length()),
      string->length());
}
