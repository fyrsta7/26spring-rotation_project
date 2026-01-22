  render(buffer.data(), &length);
  react_native_assert(length < kPropNameLengthHardCap);
  return std::string{buffer.data(), length};
}

static bool areFieldsEqual(char const *lhs, char const *rhs) {
