static bool selectorStartsWithName(ASTContext &ctx, clang::Selector sel,
                                   Identifier name) {
  return sel.getNameForSlot(0) == name.str();
}
