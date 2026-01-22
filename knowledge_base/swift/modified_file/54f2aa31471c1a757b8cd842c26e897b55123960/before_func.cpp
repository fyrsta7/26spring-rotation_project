  void getUnresolvedMemberCompletions(Type T) {
    if (!T->getNominalOrBoundGenericNominal())
      return;

    // We can only say .foo where foo is a static member of the contextual
    // type and has the same type (or if the member is a function, then the
    // same result type) as the contextual type.
    FilteredDeclConsumer consumer(*this, [=](ValueDecl *VD,
                                             DeclVisibilityKind reason) {
      if (!VD->hasInterfaceType()) {
        TypeResolver->resolveDeclSignature(VD);
        if (!VD->hasInterfaceType())
          return false;
      }

      // Enum element decls can always be referenced by implicit member
      // expression.
      if (isa<EnumElementDecl>(VD))
        return true;

      // Only non-failable constructors are implicitly referenceable.
      if (auto CD = dyn_cast<ConstructorDecl>(VD)) {
        switch (CD->getFailability()) {
          case OTK_None:
          case OTK_ImplicitlyUnwrappedOptional:
            return true;
          case OTK_Optional:
            return false;
        }
      }

      // Otherwise, check the result type matches the contextual type.
      auto declTy = getTypeOfMember(VD, T);
      if (declTy->hasError())
        return false;

      DeclContext *DC = const_cast<DeclContext *>(CurrDeclContext);

      // Member types can also be implicitly referenceable as long as it's
      // convertible to the contextual type.
      if (auto CD = dyn_cast<TypeDecl>(VD)) {
        declTy = declTy->getMetatypeInstanceType();
        return swift::isConvertibleTo(declTy, T, *DC);
      }

      // Only static member can be referenced.
      if (!VD->isStatic())
        return false;

      if (isa<FuncDecl>(VD)) {
        // Strip '(Self.Type) ->' and parameters.
        declTy = declTy->castTo<AnyFunctionType>()->getResult();
        declTy = declTy->castTo<AnyFunctionType>()->getResult();
      } else if (auto FT = declTy->getAs<AnyFunctionType>()) {
        // The compiler accepts 'static var factory: () -> T' for implicit
        // member expression.
        // FIXME: This emits just 'factory'. We should emit 'factory()' instead.
        declTy = FT->getResult();
      }
      return swift::isConvertibleTo(declTy, T, *DC);
    });

    auto baseType = MetatypeType::get(T);
    llvm::SaveAndRestore<LookupKind> SaveLook(Kind, LookupKind::ValueExpr);
    llvm::SaveAndRestore<Type> SaveType(ExprType, baseType);
    llvm::SaveAndRestore<bool> SaveUnresolved(IsUnresolvedMember, true);
    lookupVisibleMemberDecls(consumer, baseType, CurrDeclContext,
                             TypeResolver,
                             /*includeInstanceMembers=*/false);
  }
