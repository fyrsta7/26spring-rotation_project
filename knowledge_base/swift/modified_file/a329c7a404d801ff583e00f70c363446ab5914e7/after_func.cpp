static void checkRedeclaration(TypeChecker &tc, ValueDecl *current) {
  // If we've already checked this declaration, don't do it again.
  if (current->alreadyCheckedRedeclaration())
    return;

  // Make sure we don't do this checking again.
  current->setCheckedRedeclaration(true);

  // Ignore invalid declarations.
  if (current->isInvalid())
    return;

  // If this declaration isn't from a source file, don't check it.
  // FIXME: Should restrict this to the source file we care about.
  DeclContext *currentDC = current->getDeclContext();
  SourceFile *currentFile = currentDC->getParentSourceFile();
  if (!currentFile || currentDC->isLocalContext())
    return;

  // Find other potential definitions.
  SmallVector<ValueDecl *, 4> otherDefinitionsVec;
  ArrayRef<ValueDecl *> otherDefinitions;
  if (currentDC->isTypeContext()) {
    // Look within a type context.
    if (auto nominal = currentDC->getDeclaredTypeOfContext()->getAnyNominal()) {
      otherDefinitions = nominal->lookupDirect(current->getBaseName());
    }
  } else {
    // Look within a module context.
    currentFile->getParentModule()->lookupValue({ }, current->getBaseName(),
                                                NLKind::QualifiedLookup,
                                                otherDefinitionsVec);
    otherDefinitions = otherDefinitionsVec;
  }

  // Compare this signature against the signature of other
  // declarations with the same name.
  OverloadSignature currentSig = current->getOverloadSignature();
  Module *currentModule = current->getModuleContext();
  for (auto other : otherDefinitions) {
    // Skip invalid declarations and ourselves.
    if (current == other || other->isInvalid())
      continue;

    // Skip declarations in other modules.
    if (currentModule != other->getModuleContext())
      continue;

    // Don't compare methods vs. non-methods (which only happens with
    // operators).
    if (currentDC->isTypeContext() != other->getDeclContext()->isTypeContext())
      continue;

    // Validate the declaration.
    tc.validateDecl(other);
    if (other->isInvalid())
      continue;

    // Skip declarations in other files.
    // In practice, this means we will warn on a private declaration that
    // shadows a non-private one, but only in the file where the shadowing
    // happens. We will warn on conflicting non-private declarations in both
    // files.
    if (!other->isAccessibleFrom(currentDC))
      continue;

    // If there is a conflict, complain.
    if (conflicting(currentSig, other->getOverloadSignature())) {
      // If the two declarations occur in the same source file, make sure
      // we get the diagnostic ordering to be sensible.
      if (auto otherFile = other->getDeclContext()->getParentSourceFile()) {
        if (currentFile == otherFile &&
            current->getLoc().isValid() &&
            other->getLoc().isValid() &&
            tc.Context.SourceMgr.isBeforeInBuffer(current->getLoc(),
                                                  other->getLoc())) {
          std::swap(current, other);
        }
      }

      // If we're currently looking at a .sil and the conflicting declaration
      // comes from a .sib, don't error since we won't be considering the sil
      // from the .sib. So it's fine for the .sil to shadow it, since that's the
      // one we want.
      if (currentFile->Kind == SourceFileKind::SIL) {
        auto *otherFile = dyn_cast<SerializedASTFile>(
            other->getDeclContext()->getModuleScopeContext());
        if (otherFile && otherFile->isSIB())
          continue;
      }

      tc.diagnose(current, diag::invalid_redecl, current->getFullName());
      tc.diagnose(other, diag::invalid_redecl_prev, other->getFullName());

      current->setInvalid();
      if (current->hasType())
        current->overwriteType(ErrorType::get(tc.Context));
      break;
    }
  }
}
