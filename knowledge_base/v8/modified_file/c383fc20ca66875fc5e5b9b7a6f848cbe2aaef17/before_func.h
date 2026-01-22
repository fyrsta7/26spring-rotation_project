template <SearchMode search_mode, typename T>
int LinearSearch(T* array, Name* name, int len, int valid_entries,
                 int* out_insertion_index) {
  uint32_t hash = name->Hash();
  if (search_mode == ALL_ENTRIES) {
    for (int number = 0; number < len; number++) {
      int sorted_index = array->GetSortedKeyIndex(number);
      Name* entry = array->GetKey(sorted_index);
      uint32_t current_hash = entry->Hash();
      if (current_hash > hash) {
        if (out_insertion_index != NULL) *out_insertion_index = sorted_index;
        return T::kNotFound;
      }
      if (current_hash == hash && entry->Equals(name)) return sorted_index;
    }
    if (out_insertion_index != NULL) *out_insertion_index = len;
    return T::kNotFound;
  } else {
    DCHECK(len >= valid_entries);
    DCHECK_NULL(out_insertion_index);  // Not supported here.
    for (int number = 0; number < valid_entries; number++) {
      Name* entry = array->GetKey(number);
      uint32_t current_hash = entry->Hash();
      if (current_hash == hash && entry->Equals(name)) return number;
    }
    return T::kNotFound;
  }
}
