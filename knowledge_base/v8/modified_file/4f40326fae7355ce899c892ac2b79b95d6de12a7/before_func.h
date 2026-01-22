int TransitionArray::Search(String* name) {
  return internal::Search<ALL_ENTRIES>(this, name);
}
