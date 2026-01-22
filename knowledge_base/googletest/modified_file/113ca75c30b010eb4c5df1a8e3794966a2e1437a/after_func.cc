void FilePath::Normalize() {
  auto out = pathname_.begin();

  for (const char character : pathname_) {
    if (!IsPathSeparator(character)) {
      *(out++) = character;
    } else if (out == pathname_.begin() || *std::prev(out) != kPathSeparator) {
      *(out++) = kPathSeparator;
    } else {
      continue;
    }
  }

  pathname_.erase(out, pathname_.end());
}
