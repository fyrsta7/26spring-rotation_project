  for (;;) {
    auto b = cursor.peekBytes();
    if (b.empty()) {
      break;
    }
    hasher.Update(b.data(), b.size());
    cursor.skip(b.size());
  }
  uint64_t h1;
  uint64_t h2;
  hasher.Final(&h1, &h2);
  return static_cast<std::size_t>(h1);
}

ordering IOBufCompare::impl(const IOBuf& a, const IOBuf& b) const {
  io::Cursor ca(&a);
  io::Cursor cb(&b);
  for (;;) {
    auto ba = ca.peekBytes();
    auto bb = cb.peekBytes();
    if (ba.empty() && bb.empty()) {
      return ordering::eq;
    } else if (ba.empty()) {
