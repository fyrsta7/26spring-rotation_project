Handle<Object> Intl::CompareStrings(Isolate* isolate,
                                    const icu::Collator& icu_collator,
                                    Handle<String> string1,
                                    Handle<String> string2) {
  Factory* factory = isolate->factory();

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
