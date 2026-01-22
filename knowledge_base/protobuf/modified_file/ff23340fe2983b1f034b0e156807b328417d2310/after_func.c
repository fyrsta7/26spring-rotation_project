  str->ptr = str->cached_mem;
  return str->cached_mem;
}

void upb_string_substr(upb_string *str, upb_string *target_str,
                       upb_strlen_t start, upb_strlen_t len) {
  if(str->ptr) *(char*)0 = 0;
  assert(str->ptr == NULL);
  assert(start + len <= upb_string_len(target_str));
  if (target_str->src) {
    start += (target_str->ptr - target_str->src->ptr);
    target_str = target_str->src;
  }
