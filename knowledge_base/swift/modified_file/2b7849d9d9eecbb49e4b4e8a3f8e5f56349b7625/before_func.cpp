void IRGenModule::emitClangDecl(const clang::Decl *decl) {
  // Fast path for the case where `decl` doesn't contain executable code, so it
  // can't reference any other declarations that we would need to emit.
  if (getDeclWithExecutableCode(const_cast<clang::Decl *>(decl)) == nullptr) {
    ClangCodeGen->HandleTopLevelDecl(
                          clang::DeclGroupRef(const_cast<clang::Decl*>(decl)));
    return;
  }

  if (!GlobalClangDecls.insert(decl->getCanonicalDecl()).second)
    return;
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
    if (clang::Decl *executableDecl = getDeclWithExecutableCode(next)) {
        refFinder.TraverseDecl(executableDecl);
        next = executableDecl;
    }
    ClangCodeGen->HandleTopLevelDecl(clang::DeclGroupRef(next));
  }
}
