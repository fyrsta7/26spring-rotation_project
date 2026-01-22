  auto LexHorizontalWhitespace(llvm::StringRef& source_text) -> void {
    CARBON_DCHECK(source_text.front() == ' ' || source_text.front() == '\t');
    NoteWhitespace();
    // Handle adjacent whitespace quickly. This comes up frequently for example
    // due to indentation. We don't expect *huge* runs, so just use a scalar
    // loop. While still scalar, this avoids repeated table dispatch and marking
    // whitespace. We use `ssize_t` in the loop for performance.
    ssize_t ws_count = 1;
    ssize_t size = source_text.size();
    while (ws_count < size &&
           (source_text[ws_count] == ' ' || source_text[ws_count] == '\t')) {
      ++ws_count;
    }
    current_column_ += ws_count;
    source_text = source_text.drop_front(ws_count);
  }
