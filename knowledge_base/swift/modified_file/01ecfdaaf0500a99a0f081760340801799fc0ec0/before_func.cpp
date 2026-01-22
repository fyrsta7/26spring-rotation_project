void swift::
collectDefaultImplementationForProtocolMembers(ProtocolDecl *PD,
                    llvm::SmallDenseMap<ValueDecl*, ValueDecl*> &DefaultMap) {
  Type BaseTy = PD->getDeclaredInterfaceType();
  DeclContext *DC = PD->getInnermostDeclContext();
  auto HandleMembers = [&](DeclRange Members) {
    for (Decl *D : Members) {
      auto *VD = dyn_cast<ValueDecl>(D);

      // Skip non-value decl.
      if (!VD)
        continue;

      // Skip decls with empty names, e.g. setter/getters for properties.
      if (VD->getBaseName().empty())
        continue;

      ResolvedMemberResult Result = resolveValueMember(*DC, BaseTy,
                                                       VD->getFullName());
      assert(Result);
      for (auto *Default : Result.getMemberDecls(InterestedMemberKind::All)) {
        if (PD == Default->getDeclContext()->getExtendedProtocolDecl()) {
          DefaultMap.insert({Default, VD});
        }
      }
    }
  };

  // Collect the default implementations for the members in this given protocol.
  HandleMembers(PD->getMembers());

  // Collect the default implementations for the members in the inherited
  // protocols.
  for (auto *IP : PD->getInheritedProtocols())
    HandleMembers(IP->getMembers());
}
