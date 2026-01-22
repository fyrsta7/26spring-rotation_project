  str->ptr = str->cached_mem;
  return str->cached_mem;
}

void upb_string_substr(upb_string *str, upb_string *target_str,
                       upb_strlen_t start, upb_strlen_t len) {
  if(str->ptr) *(char*)0 = 0;
  assert(str->ptr == NULL);
