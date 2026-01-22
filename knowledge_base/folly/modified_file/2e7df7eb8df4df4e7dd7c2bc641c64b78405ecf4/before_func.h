/**
 * Enforce that the suffix following a number is made up only of whitespace.
 */
inline ConversionCode enforceWhitespaceErr(StringPiece sp) {
  for (auto c : sp) {
    if (UNLIKELY(!std::isspace(c))) {
      return ConversionCode::NON_WHITESPACE_AFTER_END;
    }
