static bool areFieldsEqual(char const *lhs, char const *rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return lhs == rhs;
  }
  return lhs == rhs || strcmp(lhs, rhs) == 0;
}