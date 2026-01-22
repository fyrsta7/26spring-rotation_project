static bool selectorStartsWithName(ASTContext &ctx, clang::Selector sel,
                                   Identifier name) {
  return ctx.getIdentifier(sel.getNameForSlot(0)) == name;
}
