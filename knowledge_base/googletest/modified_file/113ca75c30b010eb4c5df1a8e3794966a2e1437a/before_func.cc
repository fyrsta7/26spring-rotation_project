void FilePath::Normalize() {
  std::string normalized_pathname;
  normalized_pathname.reserve(pathname_.length());

  for (const char character : pathname_) {
    if (!IsPathSeparator(character)) {
      normalized_pathname.push_back(character);
    } else if (normalized_pathname.empty() ||
               normalized_pathname.back() != kPathSeparator) {
      normalized_pathname.push_back(kPathSeparator);
    } else {
      continue;
    }
  }

  pathname_ = normalized_pathname;
}
