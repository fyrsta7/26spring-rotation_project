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
}
