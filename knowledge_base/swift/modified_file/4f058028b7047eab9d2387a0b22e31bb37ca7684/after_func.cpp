Token Lexer::getTokenAt(SourceLoc Loc) {
  assert(BufferID == static_cast<unsigned>(
                         SourceMgr.findBufferContainingLoc(Loc)) &&
         "location from the wrong buffer");

  Lexer L(LangOpts, SourceMgr, BufferID, Diags, InSILMode,
          HashbangMode::Allowed, CommentRetentionMode::None,
          TriviaRetentionMode::WithoutTrivia);
  L.restoreState(State(Loc));
  return L.peekNextToken();
}
