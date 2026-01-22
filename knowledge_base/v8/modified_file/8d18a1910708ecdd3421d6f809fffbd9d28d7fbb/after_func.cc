Handle<Object> Intl::CompareStrings(Isolate* isolate,
                                    const icu::Collator& icu_collator,
                                    Handle<String> string1,
                                    Handle<String> string2) {
  Factory* factory = isolate->factory();

  // Early return for identical strings.
  if (string1.is_identical_to(string2)) {
    return factory->NewNumberFromInt(UCollationResult::UCOL_EQUAL);
  }

  // Early return for empty strings.
  if (string1->length() == 0) {
    return factory->NewNumberFromInt(string2->length() == 0
                                         ? UCollationResult::UCOL_EQUAL
                                         : UCollationResult::UCOL_LESS);
  }
  if (string2->length() == 0) {
    return factory->NewNumberFromInt(UCollationResult::UCOL_GREATER);
  }

  string1 = String::Flatten(isolate, string1);
  string2 = String::Flatten(isolate, string2);

  UCollationResult result;
  UErrorCode status = U_ZERO_ERROR;
  icu::UnicodeString string_val1 = Intl::ToICUUnicodeString(isolate, string1);
  icu::UnicodeString string_val2 = Intl::ToICUUnicodeString(isolate, string2);
  result = icu_collator.compare(string_val1, string_val2, status);
  DCHECK(U_SUCCESS(status));

  return factory->NewNumberFromInt(result);
}
