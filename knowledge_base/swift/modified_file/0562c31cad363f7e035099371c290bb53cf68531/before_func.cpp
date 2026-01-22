bool ConstraintSystem::resolveClosure(TypeVariableType *typeVar,
                                      Type contextualType,
                                      ConstraintLocatorBuilder locator) {
  auto *closureLocator = typeVar->getImpl().getLocator();
  auto *closure = castToExpr<ClosureExpr>(closureLocator->getAnchor());
  auto *inferredClosureType = getClosureType(closure);

  auto getContextualParamAt =
      [&contextualType, &inferredClosureType](
          unsigned index) -> Optional<AnyFunctionType::Param> {
    auto *fnType = contextualType->getAs<FunctionType>();
    if (!fnType)
      return None;

    auto numContextualParams = fnType->getNumParams();
    if (numContextualParams != inferredClosureType->getNumParams() ||
        numContextualParams <= index)
      return None;

    return fnType->getParams()[index];
  };

  // Check whether given contextual parameter type could be
  // used to bind external closure parameter type.
  auto isSuitableContextualType = [](Type contextualTy) {
    // We need to wait until contextual type
    // is fully resolved before binding it.
    if (contextualTy->isTypeVariableOrMember())
      return false;

    // If contextual type has an error, let's wait for inference,
    // otherwise contextual would interfere with diagnostics.
    if (contextualTy->hasError())
      return false;

    if (isa<TypeAliasType>(contextualTy.getPointer())) {
      auto underlyingTy = contextualTy->getDesugaredType();
      // FIXME: typealias pointing to an existential type is special
      // because if the typealias has type variables then we'd end up
      // opening existential from a type with unresolved generic
      // parameter(s), which CSApply can't currently simplify while
      // building type-checked AST because `OpenedArchetypeType` doesn't
      // propagate flags. Example is as simple as `{ $0.description }`
      // where `$0` is `Error` that inferred from a (generic) typealias.
      if (underlyingTy->isExistentialType() && contextualTy->hasTypeVariable())
        return false;
    }

    return true;
  };

  // Determine whether a result builder will be applied.
  auto resultBuilderType = getOpenedResultBuilderTypeFor(*this, locator);

  // Determine whether to introduce one-way constraints between the parameter's
  // type as seen in the body of the closure and the external parameter
  // type.
  bool oneWayConstraints =
    getASTContext().TypeCheckerOpts.EnableOneWayClosureParameters ||
    resultBuilderType;

  auto *paramList = closure->getParameters();
  SmallVector<AnyFunctionType::Param, 4> parameters;
  for (unsigned i = 0, n = paramList->size(); i != n; ++i) {
    auto param = inferredClosureType->getParams()[i];
    auto *paramDecl = paramList->get(i);

    // In case of anonymous parameters let's infer flags from context
    // that helps to infer variadic and inout earlier.
    if (closure->hasAnonymousClosureVars()) {
      if (auto contextualParam = getContextualParamAt(i))
        param = param.withFlags(contextualParam->getParameterFlags());
    }

    if (paramDecl->hasAttachedPropertyWrapper()) {
      Type backingType;
      Type wrappedValueType;

      if (paramDecl->hasImplicitPropertyWrapper()) {
        if (auto contextualType = getContextualParamAt(i)) {
          backingType = contextualType->getPlainType();
        } else {
          // There may not be a contextual parameter type if the contextual
          // type is not a function type or if closure body declares too many
          // parameters.
          auto *paramLoc =
              getConstraintLocator(closure, LocatorPathElt::TupleElement(i));
          backingType = createTypeVariable(paramLoc, TVO_CanBindToHole);
        }

        wrappedValueType = createTypeVariable(getConstraintLocator(paramDecl),
                                              TVO_CanBindToHole | TVO_CanBindToLValue);
      } else {
        auto *wrapperAttr = paramDecl->getAttachedPropertyWrappers().front();
        auto wrapperType = paramDecl->getAttachedPropertyWrapperType(0);
        backingType = replaceInferableTypesWithTypeVars(
            wrapperType, getConstraintLocator(wrapperAttr->getTypeRepr()));
        wrappedValueType = computeWrappedValueType(paramDecl, backingType);
      }

      auto *backingVar = paramDecl->getPropertyWrapperBackingProperty();
      setType(backingVar, backingType);

      auto *localWrappedVar = paramDecl->getPropertyWrapperWrappedValueVar();
      setType(localWrappedVar, wrappedValueType);

      if (auto *projection = paramDecl->getPropertyWrapperProjectionVar()) {
        setType(projection, computeProjectedValueType(paramDecl, backingType));
      }

      auto result = applyPropertyWrapperToParameter(backingType, param.getParameterType(),
                                                    paramDecl, paramDecl->getName(),
                                                    ConstraintKind::Equal,
                                                    getConstraintLocator(closure));
      if (result.isFailure())
        return false;
    }

    Type internalType;
    if (paramDecl->getTypeRepr()) {
      // Internal type is the type used in the body of the closure,
      // so "external" type translates to it as follows:
      //  - `Int...` -> `[Int]`,
      //  - `inout Int` -> `@lvalue Int`.
      internalType = param.getParameterType();

      // When there are type variables in the type and we have enabled
      // one-way constraints, create a fresh type variable to handle the
      // binding.
      if (oneWayConstraints && internalType->hasTypeVariable()) {
        auto *paramLoc =
            getConstraintLocator(closure, LocatorPathElt::TupleElement(i));
        auto *typeVar = createTypeVariable(paramLoc, TVO_CanBindToLValue |
                                                         TVO_CanBindToNoEscape);
        addConstraint(
            ConstraintKind::OneWayBindParam, typeVar, internalType, paramLoc);
        internalType = typeVar;
      }
    } else {
      auto *paramLoc =
          getConstraintLocator(closure, LocatorPathElt::TupleElement(i));

      auto *typeVar = createTypeVariable(paramLoc, TVO_CanBindToLValue |
                                                       TVO_CanBindToNoEscape);

      // If external parameter is variadic it translates into an array in
      // the body of the closure.
      internalType =
          param.isVariadic() ? ArraySliceType::get(typeVar) : Type(typeVar);

      auto externalType = param.getOldType();

      // Performance optimization.
      //
      // If there is a concrete contextual type we could use, let's bind
      // it to the external type right away because internal type has to
      // be equal to that type anyway (through `BindParam` on external type
      // i.e. <internal> bind param <external> conv <concrete contextual>).
      //
      // Note: it's correct to avoid doing this, but it would result
      // in (a lot) more checking since solver would have to re-discover,
      // re-attempt and fail parameter type while solving for overloaded
      // choices in the body.
      if (auto contextualParam = getContextualParamAt(i)) {
        auto paramTy = simplifyType(contextualParam->getOldType());
        if (isSuitableContextualType(paramTy))
          addConstraint(ConstraintKind::Bind, externalType, paramTy, paramLoc);
      }

      if (oneWayConstraints) {
        addConstraint(
            ConstraintKind::OneWayBindParam, typeVar, externalType, paramLoc);
      } else {
        addConstraint(
            ConstraintKind::BindParam, externalType, typeVar, paramLoc);
      }
    }

    setType(paramDecl, internalType);
    parameters.push_back(param);
  }

  // Propagate @Sendable from the contextual type to the closure.
  auto closureExtInfo = inferredClosureType->getExtInfo();
  if (auto contextualFnType = contextualType->getAs<FunctionType>()) {
    if (contextualFnType->isSendable())
      closureExtInfo = closureExtInfo.withConcurrent();
  }

  auto closureType =
      FunctionType::get(parameters, inferredClosureType->getResult(),
                        closureExtInfo);
  assignFixedType(typeVar, closureType, closureLocator);

  // If there is a result builder to apply, do so now.
  if (resultBuilderType) {
    if (auto result = matchResultBuilder(
            closure, resultBuilderType, closureType->getResult(),
            ConstraintKind::Conversion, locator)) {
      return result->isSuccess();
    }
  }

  // If this closure should be type-checked as part of this expression,
  // generate constraints for it now.
  auto &ctx = getASTContext();
  if (shouldTypeCheckInEnclosingExpression(closure)) {
    if (generateConstraints(closure, closureType->getResult()))
      return false;
  } else if (!hasExplicitResult(closure)) {
    // If this closure has an empty body and no explicit result type
    // let's bind result type to `Void` since that's the only type empty body
    // can produce. Otherwise, if (multi-statement) closure doesn't have
    // an explicit result (no `return` statements) let's default it to `Void`.
    auto constraintKind = (closure->hasEmptyBody() && !closure->hasExplicitResultType())
                              ? ConstraintKind::Bind
                              : ConstraintKind::Defaultable;
    addConstraint(
        constraintKind, inferredClosureType->getResult(), ctx.TheEmptyTupleType,
        getConstraintLocator(closure, ConstraintLocator::ClosureResult));
  }

  return true;
}
