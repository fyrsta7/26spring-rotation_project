bool SharedFunctionInfo::PassesFilter(const char* raw_filter) {
  // Filters are almost always "*", so check for that and exit quickly.
  if (V8_LIKELY(raw_filter[0] == '*' && raw_filter[1] == '\0')) {
    return true;
  }
  base::Vector<const char> filter = base::CStrVector(raw_filter);
  return v8::internal::PassesFilter(base::CStrVector(DebugNameCStr().get()),
                                    filter);
}
