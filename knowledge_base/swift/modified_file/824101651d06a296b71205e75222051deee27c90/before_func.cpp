Type Type::transformRec(
                    llvm::function_ref<Optional<Type>(TypeBase *)> fn) const {
  if (!isa<ParenType>(getPointer())) {
    // Transform this type node.
    if (Optional<Type> transformed = fn(getPointer()))
      return *transformed;

    // Recur.
  }

  // Recur into children of this type.
  TypeBase *base = getPointer();
  switch (base->getKind()) {
#define BUILTIN_TYPE(Id, Parent) \
case TypeKind::Id:
#define TYPE(Id, Parent)
#include "swift/AST/TypeNodes.def"
  case TypeKind::PrimaryArchetype:
  case TypeKind::OpenedArchetype:
  case TypeKind::SequenceArchetype:
  case TypeKind::Error:
  case TypeKind::Unresolved:
  case TypeKind::TypeVariable:
  case TypeKind::Placeholder:
  case TypeKind::GenericTypeParam:
  case TypeKind::SILToken:
  case TypeKind::Module:
    return *this;

  case TypeKind::Enum:
  case TypeKind::Struct:
  case TypeKind::Class:
  case TypeKind::Protocol: {
    auto nominalTy = cast<NominalType>(base);
    if (auto parentTy = nominalTy->getParent()) {
      parentTy = parentTy.transformRec(fn);
      if (!parentTy)
        return Type();

      if (parentTy.getPointer() == nominalTy->getParent().getPointer())
        return *this;

      return NominalType::get(nominalTy->getDecl(), parentTy,
                              Ptr->getASTContext());
    }

    return *this;
  }
      
  case TypeKind::SILBlockStorage: {
    auto storageTy = cast<SILBlockStorageType>(base);
    Type transCap = storageTy->getCaptureType().transformRec(fn);
    if (!transCap)
      return Type();
    CanType canTransCap = transCap->getCanonicalType();
    if (canTransCap != storageTy->getCaptureType())
      return SILBlockStorageType::get(canTransCap);
    return storageTy;
  }

  case TypeKind::SILBox: {
    bool changed = false;
    auto boxTy = cast<SILBoxType>(base);
#ifndef NDEBUG
    // This interface isn't suitable for updating the substitution map in a
    // generic SILBox.
    for (Type type : boxTy->getSubstitutions().getReplacementTypes()) {
      assert(type->isEqual(type.transformRec(fn))
             && "SILBoxType substitutions can't be transformed");
    }
#endif
    SmallVector<SILField, 4> newFields;
    auto *l = boxTy->getLayout();
    for (auto f : l->getFields()) {
      auto fieldTy = f.getLoweredType();
      auto transformed = fieldTy.transformRec(fn)->getCanonicalType();
      changed |= fieldTy != transformed;
      newFields.push_back(SILField(transformed, f.isMutable()));
    }
    boxTy = SILBoxType::get(Ptr->getASTContext(),
                            SILLayout::get(Ptr->getASTContext(),
                                           l->getGenericSignature(), newFields),
                            boxTy->getSubstitutions());
    return boxTy;
  }
  
  case TypeKind::SILFunction: {
    auto fnTy = cast<SILFunctionType>(base);
    bool changed = false;
    auto hasTypeErasedGenericClassType = [](Type ty) -> bool {
      return ty.findIf([](Type subType) -> bool {
        if (subType->isTypeErasedGenericClassType())
          return true;
        else
          return false;
      });
    };
    auto updateSubs = [&](SubstitutionMap &subs) -> bool {
      // This interface isn't suitable for updating the substitution map in a
      // substituted SILFunctionType.
      // TODO(SILFunctionType): Is it suitable for any SILFunctionType??
      SmallVector<Type, 4> newReplacements;
      for (Type type : subs.getReplacementTypes()) {
        auto transformed = type.transformRec(fn);
        assert((type->isEqual(transformed) ||
                (type->hasTypeParameter() && transformed->hasTypeParameter()) ||
                (hasTypeErasedGenericClassType(type) &&
                 hasTypeErasedGenericClassType(transformed))) &&
               "Substituted SILFunctionType can't be transformed into a "
               "concrete type");
        newReplacements.push_back(transformed->getCanonicalType());
        if (!type->isEqual(transformed))
          changed = true;
      }

      if (changed) {
        subs = SubstitutionMap::get(subs.getGenericSignature(),
                                    newReplacements,
                                    subs.getConformances());
      }

      return changed;
    };

    if (fnTy->isPolymorphic())
      return fnTy;

    if (auto subs = fnTy->getInvocationSubstitutions()) {
      if (updateSubs(subs)) {
        return fnTy->withInvocationSubstitutions(subs);
      }
      return fnTy;
    }

    if (auto subs = fnTy->getPatternSubstitutions()) {
      if (updateSubs(subs)) {
        return fnTy->withPatternSubstitutions(subs);
      }
      return fnTy;
    }

    SmallVector<SILParameterInfo, 8> transInterfaceParams;
    for (SILParameterInfo param : fnTy->getParameters()) {
      if (transformSILParameter(param, changed, fn)) return Type();
      transInterfaceParams.push_back(param);
    }

    SmallVector<SILYieldInfo, 8> transInterfaceYields;
    for (SILYieldInfo yield : fnTy->getYields()) {
      if (transformSILYield(yield, changed, fn)) return Type();
      transInterfaceYields.push_back(yield);
    }

    SmallVector<SILResultInfo, 8> transInterfaceResults;
    for (SILResultInfo result : fnTy->getResults()) {
      if (transformSILResult(result, changed, fn)) return Type();
      transInterfaceResults.push_back(result);
    }

    Optional<SILResultInfo> transErrorResult;
    if (fnTy->hasErrorResult()) {
      SILResultInfo result = fnTy->getErrorResult();
      if (transformSILResult(result, changed, fn)) return Type();
      transErrorResult = result;
    }

    if (!changed) return *this;

    return SILFunctionType::get(
        fnTy->getInvocationGenericSignature(),
        fnTy->getExtInfo(),
        fnTy->getCoroutineKind(),
        fnTy->getCalleeConvention(),
        transInterfaceParams,
        transInterfaceYields,
        transInterfaceResults,
        transErrorResult,
        SubstitutionMap(),
        SubstitutionMap(),
        Ptr->getASTContext(),
        fnTy->getWitnessMethodConformanceOrInvalid());
  }

#define REF_STORAGE(Name, ...) \
  case TypeKind::Name##Storage:
#include "swift/AST/ReferenceStorage.def"
  {
    auto storageTy = cast<ReferenceStorageType>(base);
    Type refTy = storageTy->getReferentType();
    Type substRefTy = refTy.transformRec(fn);
    if (!substRefTy)
      return Type();

    if (substRefTy.getPointer() == refTy.getPointer())
      return *this;

    return ReferenceStorageType::get(substRefTy, storageTy->getOwnership(),
                                     Ptr->getASTContext());
  }

  case TypeKind::UnboundGeneric: {
    auto unbound = cast<UnboundGenericType>(base);
    Type substParentTy;
    if (auto parentTy = unbound->getParent()) {
      substParentTy = parentTy.transformRec(fn);
      if (!substParentTy)
        return Type();

      if (substParentTy.getPointer() == parentTy.getPointer())
        return *this;

      return UnboundGenericType::get(unbound->getDecl(), substParentTy,
                                     Ptr->getASTContext());
    }

    return *this;
  }

  case TypeKind::BoundGenericClass:
  case TypeKind::BoundGenericEnum:
  case TypeKind::BoundGenericStruct: {
    auto bound = cast<BoundGenericType>(base);
    SmallVector<Type, 4> substArgs;
    bool anyChanged = false;
    Type substParentTy;
    if (auto parentTy = bound->getParent()) {
      substParentTy = parentTy.transformRec(fn);
      if (!substParentTy)
        return Type();

      if (substParentTy.getPointer() != parentTy.getPointer())
        anyChanged = true;
    }

    for (auto arg : bound->getGenericArgs()) {
      Type substArg = arg.transformRec(fn);
      if (!substArg)
        return Type();
      substArgs.push_back(substArg);
      if (substArg.getPointer() != arg.getPointer())
        anyChanged = true;
    }

    if (!anyChanged)
      return *this;

    return BoundGenericType::get(bound->getDecl(), substParentTy, substArgs);
  }
      
  case TypeKind::OpaqueTypeArchetype: {
    auto opaque = cast<OpaqueTypeArchetypeType>(base);
    if (opaque->getSubstitutions().empty())
      return *this;
    
    SmallVector<Type, 4> newSubs;
    bool anyChanged = false;
    for (auto replacement : opaque->getSubstitutions().getReplacementTypes()) {
      Type newReplacement = replacement.transformRec(fn);
      if (!newReplacement)
        return Type();
      newSubs.push_back(newReplacement);
      if (replacement.getPointer() != newReplacement.getPointer())
        anyChanged = true;
    }
    
    if (!anyChanged)
      return *this;
    
    // FIXME: This re-looks-up conformances instead of transforming them in
    // a systematic way.
    auto sig = opaque->getDecl()->getGenericSignature();
    auto newSubMap =
      SubstitutionMap::get(sig,
       [&](SubstitutableType *t) -> Type {
         auto index = sig->getGenericParamOrdinal(cast<GenericTypeParamType>(t));
         return newSubs[index];
       },
       LookUpConformanceInModule(opaque->getDecl()->getModuleContext()));
    return OpaqueTypeArchetypeType::get(opaque->getDecl(),
                                        opaque->getInterfaceType(),
                                        newSubMap);
  }

  case TypeKind::ExistentialMetatype: {
    auto meta = cast<ExistentialMetatypeType>(base);
    auto instanceTy = meta->getInstanceType().transformRec(fn);
    if (!instanceTy)
      return Type();

    if (instanceTy.getPointer() == meta->getInstanceType().getPointer())
      return *this;

    if (meta->hasRepresentation())
      return ExistentialMetatypeType::get(instanceTy,
                                          meta->getRepresentation());
    return ExistentialMetatypeType::get(instanceTy);
  }

  case TypeKind::Metatype: {
    auto meta = cast<MetatypeType>(base);
    auto instanceTy = meta->getInstanceType().transformRec(fn);
    if (!instanceTy)
      return Type();

    if (instanceTy.getPointer() == meta->getInstanceType().getPointer())
      return *this;

    if (meta->hasRepresentation())
      return MetatypeType::get(instanceTy, meta->getRepresentation());
    return MetatypeType::get(instanceTy);
  }

  case TypeKind::DynamicSelf: {
    auto dynamicSelf = cast<DynamicSelfType>(base);
    auto selfTy = dynamicSelf->getSelfType().transformRec(fn);
    if (!selfTy)
      return Type();

    if (selfTy.getPointer() == dynamicSelf->getSelfType().getPointer())
      return *this;

    return DynamicSelfType::get(selfTy, selfTy->getASTContext());
  }

  case TypeKind::TypeAlias: {
    auto alias = cast<TypeAliasType>(base);
    Type oldUnderlyingType = Type(alias->getSinglyDesugaredType());
    Type newUnderlyingType = oldUnderlyingType.transformRec(fn);
    if (!newUnderlyingType) return Type();

    Type oldParentType = alias->getParent();
    Type newParentType;
    if (oldParentType) {
      newParentType = oldParentType.transformRec(fn);
      if (!newParentType) return newUnderlyingType;
    }

    auto subMap = alias->getSubstitutionMap();
    for (Type oldReplacementType : subMap.getReplacementTypes()) {
      Type newReplacementType = oldReplacementType.transformRec(fn);
      if (!newReplacementType)
        return newUnderlyingType;

      // If anything changed with the replacement type, we lose the sugar.
      // FIXME: This is really unfortunate.
      if (newReplacementType.getPointer() != oldReplacementType.getPointer())
        return newUnderlyingType;
    }

    if (oldParentType.getPointer() == newParentType.getPointer() &&
        oldUnderlyingType.getPointer() == newUnderlyingType.getPointer())
      return *this;

    return TypeAliasType::get(alias->getDecl(), newParentType, subMap,
                              newUnderlyingType);
  }

  case TypeKind::Paren: {
    auto paren = cast<ParenType>(base);
    Type underlying = paren->getUnderlyingType().transformRec(fn);
    if (!underlying)
      return Type();

    if (underlying.getPointer() == paren->getUnderlyingType().getPointer())
      return *this;

    auto otherFlags = paren->getParameterFlags().withInOut(underlying->is<InOutType>());
    return ParenType::get(Ptr->getASTContext(), underlying->getInOutObjectType(), otherFlags);
  }

  case TypeKind::Pack: {
    auto pack = cast<PackType>(base);
    bool anyChanged = false;
    SmallVector<Type, 4> elements;
    unsigned Index = 0;
    for (Type eltTy : pack->getElementTypes()) {
      Type transformedEltTy = eltTy.transformRec(fn);
      if (!transformedEltTy)
        return Type();

      // If nothing has changed, just keep going.
      if (!anyChanged &&
          transformedEltTy.getPointer() == eltTy.getPointer()) {
        ++Index;
        continue;
      }

      // If this is the first change we've seen, copy all of the previous
      // elements.
      if (!anyChanged) {
        // Copy all of the previous elements.
        elements.append(pack->getElementTypes().begin(),
                        pack->getElementTypes().begin() + Index);
        anyChanged = true;
      }

      elements.push_back(transformedEltTy);
      ++Index;
    }

    if (!anyChanged)
      return *this;

    return PackType::get(Ptr->getASTContext(), elements);
  }

  case TypeKind::PackExpansion: {
    auto expand = cast<PackExpansionType>(base);
    struct ExpansionGatherer {
      llvm::function_ref<Optional<Type>(TypeBase *)> baselineFn;
      llvm::DenseMap<TypeBase *, PackType *> cache;
      unsigned maxArity;

    public:
      ExpansionGatherer(
          llvm::function_ref<Optional<Type>(TypeBase *)> baselineFn)
          : baselineFn(baselineFn), maxArity(0) {}

      Optional<Type> operator()(TypeBase *input) {
        auto remap = baselineFn(input);
        if (!remap) {
          return remap;
        }

        if (input->is<TypeVariableType>()) {
          if (auto *PT = (*remap)->getAs<PackType>()) {
            maxArity = std::max(maxArity, PT->getNumElements());
            cache.insert({input, PT});
          }
        } else if (input->isTypeSequenceParameter()) {
          if (auto *PT = (*remap)->getAs<PackType>()) {
            maxArity = std::max(maxArity, PT->getNumElements());
            cache.insert({input, PT});
          }
        }
        return remap;
      }

      std::pair<llvm::DenseMap<TypeBase *, PackType *>, unsigned>
      intoExpansions() && {
        return std::make_pair(cache, maxArity);
      }
    };

    // First, substitute down the pattern type to gather the mapping from
    // contained substitutable types to packs.
    auto gather = ExpansionGatherer{fn};
    Type transformedPat = expand->getPatternType().transformRec(gather);
    if (!transformedPat)
      return Type();

    if (transformedPat.getPointer() == expand->getPatternType().getPointer())
      return *this;

    llvm::DenseMap<TypeBase *, PackType *> expansions;
    unsigned arity;
    std::tie(expansions, arity) = std::move(gather).intoExpansions();
    if (expansions.empty()) {
      // If we didn't find any expansions, either the caller wasn't interested
      // in expanding this pack, or something has gone wrong. Leave off the
      // expansion and return the transformed type.
      return PackExpansionType::get(transformedPat);
    }

    SmallVector<Type, 8> elts;
    elts.reserve(arity);
    // Perform the expansion element-wise according to the maximum arity we
    // picked up during the gather step above.
    //
    // For a pack expansion (F<... T..., U..., ...>) and mapping
    //
    //   T... -> <X, Y, Z>
    //   U... -> <A, B, C>
    //
    // The expected expansion is
    //
    // <F<... X, A, ...>, F<... Y, B, ...>, F<... Z, C, ...> ...>
    for (unsigned i = 0; i < arity; ++i) {
      struct ElementExpander {
        const llvm::DenseMap<TypeBase *, PackType *> &expansions;
        llvm::function_ref<Optional<Type>(TypeBase *)> outerFn;
        unsigned index;

      public:
        Optional<Type> operator()(TypeBase *input) {
          // FIXME: Does this need to do bounds checking?
          if (PackType *element = expansions.lookup(input))
            return element->getElementType(index);
          return outerFn(input);
        }
      };

      auto expandedElt = expand->getPatternType().transformRec(
          ElementExpander{expansions, fn, i});
      if (!expandedElt)
        return Type();

      elts.push_back(expandedElt);
    }
    return PackType::get(base->getASTContext(), elts);
  }

  case TypeKind::Tuple: {
    auto tuple = cast<TupleType>(base);
    bool anyChanged = false;
    SmallVector<TupleTypeElt, 4> elements;
    unsigned Index = 0;
    for (const auto &elt : tuple->getElements()) {
      Type eltTy = elt.getType();
      Type transformedEltTy = eltTy.transformRec(fn);
      if (!transformedEltTy)
        return Type();

      // If nothing has changed, just keep going.
      if (!anyChanged &&
          transformedEltTy.getPointer() == elt.getType().getPointer()) {
        ++Index;
        continue;
      }

      // If this is the first change we've seen, copy all of the previous
      // elements.
      if (!anyChanged) {
        // Copy all of the previous elements.
        elements.append(tuple->getElements().begin(),
                        tuple->getElements().begin() + Index);
        anyChanged = true;
      }

      if (eltTy->isTypeSequenceParameter() &&
          transformedEltTy->is<PackType>()) {
        assert(anyChanged);
        // Splat the tuple in by copying in all of the transformed elements.
        auto tuple = dyn_cast<PackType>(transformedEltTy.getPointer());
        elements.append(tuple->getElementTypes().begin(),
                        tuple->getElementTypes().end());
      } else {
        // Add the new tuple element, with the transformed type.
        elements.push_back(elt.getWithType(transformedEltTy));
        ++Index;
      }
    }

    if (!anyChanged)
      return *this;

    return TupleType::get(elements, Ptr->getASTContext());
  }


  case TypeKind::DependentMember: {
    auto dependent = cast<DependentMemberType>(base);
    auto dependentBase = dependent->getBase().transformRec(fn);
    if (!dependentBase)
      return Type();

    if (dependentBase.getPointer() == dependent->getBase().getPointer())
      return *this;

    if (auto assocType = dependent->getAssocType())
      return DependentMemberType::get(dependentBase, assocType);

    return DependentMemberType::get(dependentBase, dependent->getName());
  }

  case TypeKind::GenericFunction:
  case TypeKind::Function: {
    auto function = cast<AnyFunctionType>(base);

    bool isUnchanged = true;

    // Transform function parameter types.
    SmallVector<AnyFunctionType::Param, 8> substParams;
    for (auto param : function->getParams()) {
      auto type = param.getPlainType();
      auto label = param.getLabel();
      auto flags = param.getParameterFlags();
      auto internalLabel = param.getInternalLabel();

      auto substType = type.transformRec(fn);
      if (!substType)
        return Type();

      if (type.getPointer() != substType.getPointer())
        isUnchanged = false;

      // FIXME: Remove this once we get rid of TVO_CanBindToInOut;
      // the only time we end up here is when the constraint solver
      // simplifies a type containing a type variable fixed to an
      // InOutType.
      if (substType->is<InOutType>()) {
        assert(flags.getValueOwnership() == ValueOwnership::Default);
        substType = substType->getInOutObjectType();
        flags = flags.withInOut(true);
      }

      substParams.emplace_back(substType, label, flags, internalLabel);
    }

    // Transform result type.
    auto resultTy = function->getResult().transformRec(fn);
    if (!resultTy)
      return Type();

    if (resultTy.getPointer() != function->getResult().getPointer())
      isUnchanged = false;

    // Transform the global actor.
    Type globalActorType;
    if (Type origGlobalActorType = function->getGlobalActor()) {
      globalActorType = origGlobalActorType.transformRec(fn);
      if (!globalActorType)
        return Type();

      if (globalActorType.getPointer() != origGlobalActorType.getPointer())
        isUnchanged = false;
    }

    if (auto genericFnType = dyn_cast<GenericFunctionType>(base)) {
#ifndef NDEBUG
      // Check that generic parameters won't be trasnformed.
      // Transform generic parameters.
      for (auto param : genericFnType->getGenericParams()) {
        assert(Type(param).transformRec(fn).getPointer() == param &&
               "GenericFunctionType transform() changes type parameter");
      }
#endif

      if (isUnchanged) return *this;

      auto genericSig = genericFnType->getGenericSignature();
      if (!function->hasExtInfo())
        return GenericFunctionType::get(genericSig, substParams, resultTy);
      return GenericFunctionType::get(genericSig, substParams, resultTy,
                                      function->getExtInfo()
                                          .withGlobalActor(globalActorType));
    }

    if (isUnchanged) return *this;

    if (!function->hasExtInfo())
      return FunctionType::get(substParams, resultTy);
    return FunctionType::get(substParams, resultTy,
                             function->getExtInfo()
                                 .withGlobalActor(globalActorType));
  }

  case TypeKind::ArraySlice: {
    auto slice = cast<ArraySliceType>(base);
    auto baseTy = slice->getBaseType().transformRec(fn);
    if (!baseTy)
      return Type();

    if (baseTy.getPointer() == slice->getBaseType().getPointer())
      return *this;

    return ArraySliceType::get(baseTy);
  }

  case TypeKind::Optional: {
    auto optional = cast<OptionalType>(base);
    auto baseTy = optional->getBaseType().transformRec(fn);
    if (!baseTy)
      return Type();

    if (baseTy.getPointer() == optional->getBaseType().getPointer())
      return *this;

    return OptionalType::get(baseTy);
  }

  case TypeKind::VariadicSequence: {
    auto seq = cast<VariadicSequenceType>(base);
    auto baseTy = seq->getBaseType().transformRec(fn);
    if (!baseTy)
      return Type();

    if (baseTy.getPointer() == seq->getBaseType().getPointer())
      return *this;

    return VariadicSequenceType::get(baseTy);
  }

  case TypeKind::Dictionary: {
    auto dict = cast<DictionaryType>(base);
    auto keyTy = dict->getKeyType().transformRec(fn);
    if (!keyTy)
      return Type();

    auto valueTy = dict->getValueType().transformRec(fn);
    if (!valueTy)
      return Type();

    if (keyTy.getPointer() == dict->getKeyType().getPointer() &&
        valueTy.getPointer() == dict->getValueType().getPointer())
      return *this;

    return DictionaryType::get(keyTy, valueTy);
  }

  case TypeKind::LValue: {
    auto lvalue = cast<LValueType>(base);
    auto objectTy = lvalue->getObjectType().transformRec(fn);
    if (!objectTy || objectTy->hasError())
      return objectTy;

    return objectTy.getPointer() == lvalue->getObjectType().getPointer() ?
      *this : LValueType::get(objectTy);
  }

  case TypeKind::InOut: {
    auto inout = cast<InOutType>(base);
    auto objectTy = inout->getObjectType().transformRec(fn);
    if (!objectTy || objectTy->hasError())
      return objectTy;
    
    return objectTy.getPointer() == inout->getObjectType().getPointer() ?
      *this : InOutType::get(objectTy);
  }

  case TypeKind::Existential: {
    auto *existential = cast<ExistentialType>(base);
    auto constraint = existential->getConstraintType().transformRec(fn);
    if (!constraint || constraint->hasError())
      return constraint;

    if (constraint.getPointer() ==
        existential->getConstraintType().getPointer())
      return *this;

    return ExistentialType::get(constraint);
  }

  case TypeKind::ProtocolComposition: {
    auto pc = cast<ProtocolCompositionType>(base);
    SmallVector<Type, 4> substMembers;
    auto members = pc->getMembers();
    bool anyChanged = false;
    unsigned index = 0;
    for (auto member : members) {
      auto substMember = member.transformRec(fn);
      if (!substMember)
        return Type();
      
      if (anyChanged) {
        substMembers.push_back(substMember);
        ++index;
        continue;
      }
      
      if (substMember.getPointer() != member.getPointer()) {
        anyChanged = true;
        substMembers.append(members.begin(), members.begin() + index);
        substMembers.push_back(substMember);
      }
      
      ++index;
    }
    
    if (!anyChanged)
      return *this;
    
    return ProtocolCompositionType::get(Ptr->getASTContext(),
                                        substMembers,
                                        pc->hasExplicitAnyObject());
  }
  }
  
  llvm_unreachable("Unhandled type in transformation");
}
