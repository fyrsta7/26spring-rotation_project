SmallVector<AssociatedTypeDecl *, 1>
AssociatedTypeDecl::getOverriddenDecls() const {
  SmallVector<AssociatedTypeDecl *, 1> assocTypes;
  for (auto decl : AbstractTypeParamDecl::getOverriddenDecls()) {
    assocTypes.push_back(cast<AssociatedTypeDecl>(decl));
  }
  return assocTypes;
}
