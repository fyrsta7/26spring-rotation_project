tok Lexer::kindOfIdentifier(StringRef Str, bool InSILMode) {
#define SIL_KEYWORD(kw)
#define KEYWORD(kw) if (Str == #kw) return tok::kw_##kw;
#include "swift/Syntax/TokenKinds.def"

  // SIL keywords are only active in SIL mode.
  if (InSILMode) {
#define SIL_KEYWORD(kw) if (Str == #kw) return tok::kw_##kw;
#include "swift/Syntax/TokenKinds.def"
  }
  return tok::identifier;
}
