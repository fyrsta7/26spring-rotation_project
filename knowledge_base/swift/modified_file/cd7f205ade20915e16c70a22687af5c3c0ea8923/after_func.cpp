void IRGenModule::emitClangDecl(const clang::Decl *decl) {
  // Ignore this decl if we've seen it before.
  if (!GlobalClangDecls.insert(decl->getCanonicalDecl()).second)
    return;

  auto valueDecl = dyn_cast<clang::ValueDecl>(decl);
  if (!valueDecl || valueDecl->isExternallyVisible()) {
    ClangCodeGen->HandleTopLevelDecl(
                          clang::DeclGroupRef(const_cast<clang::Decl*>(decl)));
    return;
  }

  SmallVector<const clang::Decl *, 8> stack;
  stack.push_back(decl);

  ClangDeclRefFinder refFinder([&](const clang::DeclRefExpr *DRE) {
    const clang::Decl *D = DRE->getDecl();
    // Check that this is a file-level declaration and not inside a function.
    // If it's a member of a file-level decl, like a C++ static member variable,
    // we want to add the entire file-level declaration because Clang doesn't
    // expect to see members directly here.
    for (auto *DC = D->getDeclContext();; DC = DC->getParent()) {
      if (DC->isFunctionOrMethod())
        return;
      if (DC->isFileContext())
        break;
      D = cast<const clang::Decl>(DC);
    }
    if (!GlobalClangDecls.insert(D->getCanonicalDecl()).second)
      return;
    stack.push_back(D);
  });

  while (!stack.empty()) {
    auto *next = const_cast<clang::Decl *>(stack.pop_back_val());
    if (auto fn = dyn_cast<clang::FunctionDecl>(next)) {
      const clang::FunctionDecl *definition;
      if (fn->hasBody(definition)) {
        refFinder.TraverseDecl(const_cast<clang::FunctionDecl *>(definition));
        next = const_cast<clang::FunctionDecl *>(definition);
      }
    }
    ClangCodeGen->HandleTopLevelDecl(clang::DeclGroupRef(next));
  }
}
