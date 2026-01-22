void SwiftLookupTableWriter::populateTableWithDecl(SwiftLookupTable &table,
                                                   NameImporter &nameImporter,
                                                   clang::Decl *decl) {
  // Skip anything from an AST file.
  if (decl->isFromASTFile())
    return;

  // Iterate into extern "C" {} type declarations.
  if (auto linkageDecl = dyn_cast<clang::LinkageSpecDecl>(decl)) {
    for (auto *decl : linkageDecl->noload_decls()) {
      populateTableWithDecl(table, nameImporter, decl);
    }
    return;
  }

  // Skip non-named declarations.
  auto named = dyn_cast<clang::NamedDecl>(decl);
  if (!named)
    return;

  // Add this entry to the lookup table.
  addEntryToLookupTable(table, named, nameImporter);
  if (auto typedefDecl = dyn_cast<clang::TypedefNameDecl>(named)) {
    if (auto typedefType = dyn_cast<clang::TemplateSpecializationType>(
            typedefDecl->getUnderlyingType())) {
      if (auto CTSD = dyn_cast<clang::ClassTemplateSpecializationDecl>(
              typedefType->getAsTagDecl())) {
        // Adding template instantiation behind typedef as a top-level entry
        // so the instantiation appears in the API.
        assert(!isa<clang::ClassTemplatePartialSpecializationDecl>(CTSD) &&
            "Class template partial specialization cannot appear behind typedef");
        addEntryToLookupTable(table, CTSD, nameImporter);
      }
    }
  }
}
