Type TypeChecker::lookupBoolType(const DeclContext *dc) {
  if (!boolType) {
    boolType = ([&] {
      SmallVector<ValueDecl *, 2> results;
      getStdlibModule(dc)->lookupValue({}, Context.getIdentifier("Bool"),
                                       NLKind::QualifiedLookup, results);
      if (results.size() != 1) {
        diagnose(SourceLoc(), diag::bool_type_broken);
        return Type();
      }

      auto tyDecl = dyn_cast<TypeDecl>(results.front());
      if (!tyDecl) {
        diagnose(SourceLoc(), diag::bool_type_broken);
        return Type();
      }

      return tyDecl->getDeclaredType();
    })();
  }
  return *boolType;
}
