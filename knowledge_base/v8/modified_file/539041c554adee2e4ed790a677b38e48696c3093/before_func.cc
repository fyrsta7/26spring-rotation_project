Node* GraphAssembler::TaggedEqual(Node* left, Node* right) {
  if (machine()->Is64() && COMPRESS_POINTERS_BOOL) {
    // Allow implicit truncation.
    return Word32Equal(left, right);
  }
  return WordEqual(left, right);
}
