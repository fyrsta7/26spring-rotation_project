static bool areFieldsEqual(char const *lhs, char const *rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return lhs == rhs;
  }
  return std::string(lhs) == std::string(rhs);
}