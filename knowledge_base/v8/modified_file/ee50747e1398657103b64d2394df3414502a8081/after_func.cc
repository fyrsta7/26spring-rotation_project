BUILTIN(StringPrototypeEndsWith) {
  HandleScope handle_scope(isolate);
  TO_THIS_STRING(str, "String.prototype.endsWith");

  // Check if the search string is a regExp and fail if it is.
  Handle<Object> search = args.atOrUndefined(isolate, 1);
  Maybe<bool> is_reg_exp = RegExpUtils::IsRegExp(isolate, search);
  if (is_reg_exp.IsNothing()) {
    DCHECK(isolate->has_pending_exception());
    return isolate->heap()->exception();
  }
  if (is_reg_exp.FromJust()) {
    THROW_NEW_ERROR_RETURN_FAILURE(
        isolate, NewTypeError(MessageTemplate::kFirstArgumentNotRegExp,
                              isolate->factory()->NewStringFromStaticChars(
                                  "String.prototype.endsWith")));
  }
  Handle<String> search_string;
  ASSIGN_RETURN_FAILURE_ON_EXCEPTION(isolate, search_string,
                                     Object::ToString(isolate, search));

  Handle<Object> position = args.atOrUndefined(isolate, 2);
  int end;

  if (position->IsUndefined(isolate)) {
    end = str->length();
  } else {
    ASSIGN_RETURN_FAILURE_ON_EXCEPTION(isolate, position,
                                       Object::ToInteger(isolate, position));
    double index = std::max(position->Number(), 0.0);
    index = std::min(index, static_cast<double>(str->length()));
    end = static_cast<uint32_t>(index);
  }

  int start = end - search_string->length();
  if (start < 0) return isolate->heap()->false_value();

  str = String::Flatten(str);
  search_string = String::Flatten(search_string);

  DisallowHeapAllocation no_gc;  // ensure vectors stay valid
  String::FlatContent str_content = str->GetFlatContent();
  String::FlatContent search_content = search_string->GetFlatContent();

  if (str_content.IsOneByte() && search_content.IsOneByte()) {
    Vector<const uint8_t> str_vector = str_content.ToOneByteVector();
    Vector<const uint8_t> search_vector = search_content.ToOneByteVector();

    return isolate->heap()->ToBoolean(memcmp(str_vector.start() + start,
                                             search_vector.start(),
                                             search_string->length()) == 0);
  }

  FlatStringReader str_reader(isolate, str);
  FlatStringReader search_reader(isolate, search_string);

  for (int i = 0; i < search_string->length(); i++) {
    if (str_reader.Get(start + i) != search_reader.Get(i)) {
      return isolate->heap()->false_value();
    }
  }
  return isolate->heap()->true_value();
}
