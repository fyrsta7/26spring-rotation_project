syntax::RawSyntaxInfo Lexer::fullLex() {
  if (NextToken.isEscapedIdentifier()) {
    LeadingTrivia.push_back(syntax::TriviaPiece::backtick());
    TrailingTrivia.insert(TrailingTrivia.begin(),
                          syntax::TriviaPiece::backtick());
  }
  auto Loc = NextToken.getLoc();
  auto Result = syntax::RawTokenSyntax::make(NextToken.getKind(),
                                        OwnedString(NextToken.getText()).copy(),
                                        syntax::SourcePresence::Present,
                                        {LeadingTrivia}, {TrailingTrivia});
  LeadingTrivia.clear();
  TrailingTrivia.clear();
  if (NextToken.isNot(tok::eof)) {
    lexImpl();
  }
  return {Loc, Result};
}
