tok Lexer::kindOfIdentifier(StringRef Str, bool InSILMode) {
  tok Kind = llvm::StringSwitch<tok>(Str)
#define KEYWORD(kw) \
    .Case(#kw, tok::kw_##kw)
#include "swift/Syntax/TokenKinds.def"
    .Default(tok::identifier);

  // SIL keywords are only active in SIL mode.
  switch (Kind) {
#define SIL_KEYWORD(kw) \
    case tok::kw_##kw:
#include "swift/Syntax/TokenKinds.def"
      if (!InSILMode)
        Kind = tok::identifier;
      break;
    default:
      break;
  }
  return Kind;
}
