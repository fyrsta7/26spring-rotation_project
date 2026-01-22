SmallVector<AssociatedTypeDecl *, 1>
AssociatedTypeDecl::getOverriddenDecls() const {
  // FIXME: Performance hack because we end up looking at the overridden
  // declarations of an associated type a *lot*.
  OverriddenDeclsRequest request{const_cast<AssociatedTypeDecl *>(this)};
  SmallVector<ValueDecl *, 1> overridden;
  if (auto cached = request.getCachedResult())
    overridden = std::move(*cached);
  else
    overridden = AbstractTypeParamDecl::getOverriddenDecls();

  SmallVector<AssociatedTypeDecl *, 1> assocTypes;
  for (auto decl : overridden) {
    assocTypes.push_back(cast<AssociatedTypeDecl>(decl));
  }
  return assocTypes;
}
