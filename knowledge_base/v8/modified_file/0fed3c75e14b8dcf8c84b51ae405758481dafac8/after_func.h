template <typename PatternChar, typename SubjectChar>
inline int FindFirstCharacter(Vector<const PatternChar> pattern,
                              Vector<const SubjectChar> subject, int index) {
  const PatternChar pattern_first_char = pattern[0];
  const int max_n = (subject.length() - pattern.length() + 1);

  if (sizeof(SubjectChar) == 2 && pattern_first_char == 0) {
    // Special-case looking for the 0 char in other than one-byte strings.
    // memchr mostly fails in this case due to every other byte being 0 in text
    // that is mostly ascii characters.
    for (int i = index; i < max_n; ++i) {
      if (subject[i] == 0) return i;
    }
    return -1;
  }
  const uint8_t search_byte = GetHighestValueByte(pattern_first_char);
  const SubjectChar search_char = static_cast<SubjectChar>(pattern_first_char);
  int pos = index;
  do {
    DCHECK_GE(max_n - pos, 0);
    const SubjectChar* char_pos = reinterpret_cast<const SubjectChar*>(
        memchr(subject.begin() + pos, search_byte,
               (max_n - pos) * sizeof(SubjectChar)));
    if (char_pos == nullptr) return -1;
    char_pos = AlignDown(char_pos, sizeof(SubjectChar));
    pos = static_cast<int>(char_pos - subject.begin());
    if (subject[pos] == search_char) return pos;
  } while (++pos < max_n);

  return -1;
}
