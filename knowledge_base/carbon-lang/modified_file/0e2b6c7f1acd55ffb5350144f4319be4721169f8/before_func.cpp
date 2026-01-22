  auto LexHorizontalWhitespace(llvm::StringRef& source_text) -> void {
    CARBON_DCHECK(source_text.front() == ' ' || source_text.front() == '\t');
    NoteWhitespace();
    ++current_column_;
    source_text = source_text.drop_front();
  }
