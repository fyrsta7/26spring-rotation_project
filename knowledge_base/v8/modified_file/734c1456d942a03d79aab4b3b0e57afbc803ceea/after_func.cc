icu::UnicodeString Intl::ToICUUnicodeString(Isolate* isolate,
                                            Handle<String> string) {
  DCHECK(string->IsFlat());
  DisallowHeapAllocation no_gc;
  std::unique_ptr<uc16[]> sap;
  // Short one-byte strings can be expanded on the stack to avoid allocating a
  // temporary buffer.
  constexpr int kShortStringSize = 80;
  UChar short_string_buffer[kShortStringSize];
  const UChar* uchar_buffer = nullptr;
  const String::FlatContent& flat = string->GetFlatContent(no_gc);
  int32_t length = string->length();
  if (flat.IsOneByte() && length <= kShortStringSize) {
    CopyChars(short_string_buffer, flat.ToOneByteVector().begin(), length);
    uchar_buffer = short_string_buffer;
  } else {
    uchar_buffer = GetUCharBufferFromFlat(flat, &sap, length);
  }
  return icu::UnicodeString(uchar_buffer, length);
}
