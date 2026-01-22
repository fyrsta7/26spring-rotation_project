Type TypeChecker::lookupBoolType(const DeclContext *dc) {
  if (!boolType) {
    boolType = ([&] {
      UnqualifiedLookup boolLookup(Context.getIdentifier("Bool"),
                                   getStdlibModule(dc), nullptr,
                                   SourceLoc(),
                                   /*IsTypeLookup=*/true);
      if (!boolLookup.isSuccess()) {
        diagnose(SourceLoc(), diag::bool_type_broken);
        return Type();
      }
      TypeDecl *tyDecl = boolLookup.getSingleTypeResult();

      if (!tyDecl) {
        diagnose(SourceLoc(), diag::bool_type_broken);
        return Type();
      }

      return tyDecl->getDeclaredType();
    })();
  }
  return *boolType;
}
