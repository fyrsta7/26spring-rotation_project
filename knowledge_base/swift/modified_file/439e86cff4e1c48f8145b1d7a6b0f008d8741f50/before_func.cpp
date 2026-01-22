ConstraintSystem::SolutionKind
ConstraintSystem::matchTypes(Type type1, Type type2, TypeMatchKind kind,
                             unsigned flags, bool &trivial) {
  // Desugar both types.
  auto desugar1 = type1->getDesugaredType();
  auto desugar2 = type2->getDesugaredType();

  // If we have type variables that have been bound to fixed types, look through
  // to the fixed type.
  auto typeVar1 = dyn_cast<TypeVariableType>(desugar1);
  if (typeVar1) {
    if (auto fixed = getFixedType(typeVar1)) {
      type1 = fixed;
      desugar1 = fixed->getDesugaredType();
      typeVar1 = nullptr;
    }
  }

  auto typeVar2 = dyn_cast<TypeVariableType>(desugar2);
  if (typeVar2) {
    if (auto fixed = getFixedType(typeVar2)) {
      type2 = fixed;
      desugar2 = fixed->getDesugaredType();
      typeVar2 = nullptr;
    }
  }

  // If we have a same-type-as-rvalue constraint, and the right-hand side
  // has a form that is either definitely an lvalue or definitely an rvalue,
  // force the right-hand side to be an rvalue and
  if (kind == TypeMatchKind::SameTypeRvalue) {
    if (isa<LValueType>(desugar2)) {
      // The right-hand side is an lvalue type. Strip off the lvalue and
      // call this a normal 'same-type' constraint.
      type2 = type2->castTo<LValueType>()->getObjectType();
      desugar2 = type2->getDesugaredType();
      kind = TypeMatchKind::SameType;
      flags |= TMF_GenerateConstraints;
    } else if (!type2->is<TypeVariableType>()) {
      // The right-hand side is guaranteed to be an rvalue type. Call this
      // a normal same-type constraint.
      kind = TypeMatchKind::SameType;
      flags |= TMF_GenerateConstraints;
    }

    if (auto desugarFunc2 = dyn_cast<FunctionType>(desugar2)) {
      // The right-hand side is a function type, which is guaranteed to be
      // an rvalue type. Call this a normal same-type constraint, and
      // strip off the [auto_closure], which is not part of the type.
      if (desugarFunc2->isAutoClosure()) {
        auto func2 = type2->castTo<FunctionType>();
        type2 = FunctionType::get(func2->getInput(), func2->getResult(),
                                  TC.Context);
        desugar2 = type2.getPointer();
      }
      kind = TypeMatchKind::SameType;
      flags |= TMF_GenerateConstraints;
    }
  }

  // If either (or both) types are type variables, unify the type variables.
  if (typeVar1 || typeVar2) {
    switch (kind) {
    case TypeMatchKind::BindType:
    case TypeMatchKind::SameType: {
      if (typeVar1 && typeVar2) {
        auto rep1 = getRepresentative(typeVar1);
        auto rep2 = getRepresentative(typeVar2);
        if (rep1 == rep2) {
          // We already merged these two types, so this constraint is
          // trivially solved.
          return SolutionKind::TriviallySolved;
        }

        // Merge the equivalence classes corresponding to these two variables.
        mergeEquivalenceClasses(rep1, rep2);
        return SolutionKind::Solved;
      }

      // Provide a fixed type for the type variable.
      bool wantRvalue = kind == TypeMatchKind::SameType;
      if (typeVar1)
        assignFixedType(typeVar1, wantRvalue ? type2->getRValueType() : type2);
      else
        assignFixedType(typeVar2, wantRvalue ? type1->getRValueType() : type1);
      return SolutionKind::Solved;
    }

    case TypeMatchKind::SameTypeRvalue:
    case TypeMatchKind::TrivialSubtype:
    case TypeMatchKind::Subtype:
    case TypeMatchKind::Conversion:
    case TypeMatchKind::Construction:
      if (flags & TMF_GenerateConstraints) {
        // Add a new constraint between these types. We consider the current
        // type-matching problem to the "solved" by this addition, because
        // this new constraint will be solved at a later point.
        // Obviously, this must not happen at the top level, or the algorithm
        // would not terminate.
        addConstraint(getConstraintKind(kind), type1, type2);
        return SolutionKind::Solved;
      }

      // We couldn't solve this constraint. If only one of the types is a type
      // variable, perhaps we can do something with it below.
      if (typeVar1 && typeVar2)
        return typeVar1 == typeVar2? SolutionKind::TriviallySolved
                                   : SolutionKind::Unsolved;
        
      break;
    }
  }

  // Decompose parallel structure.
  unsigned subFlags = flags | TMF_GenerateConstraints;
  if (desugar1->getKind() == desugar2->getKind()) {
    switch (desugar1->getKind()) {
#define SUGARED_TYPE(id, parent) case TypeKind::id:
#define TYPE(id, parent)
#include "swift/AST/TypeNodes.def"
      llvm_unreachable("Type has not been desugared completely");

#define ALWAYS_CANONICAL_TYPE(id, parent) case TypeKind::id:
#define TYPE(id, parent)
#include "swift/AST/TypeNodes.def"
        return desugar1 == desugar2
                 ? SolutionKind::TriviallySolved
                 : SolutionKind::Error;

    case TypeKind::Error:
      return SolutionKind::Error;

    case TypeKind::UnstructuredUnresolved:
      llvm_unreachable("Unstructured unresolved type");

    case TypeKind::TypeVariable:
      llvm_unreachable("Type variables handled above");

    case TypeKind::Tuple: {
      auto tuple1 = cast<TupleType>(desugar1);
      auto tuple2 = cast<TupleType>(desugar2);
      return matchTupleTypes(tuple1, tuple2, kind, flags, trivial);
    }

    case TypeKind::OneOf:
    case TypeKind::Struct:
    case TypeKind::Class: {
      auto nominal1 = cast<NominalType>(desugar1);
      auto nominal2 = cast<NominalType>(desugar2);
      if (nominal1->getDecl() == nominal2->getDecl()) {
        assert((bool)nominal1->getParent() == (bool)nominal2->getParent() &&
               "Mismatched parents of nominal types");

        if (!nominal1->getParent())
          return SolutionKind::TriviallySolved;

        // Match up the parents, exactly.
        return matchTypes(nominal1->getParent(), nominal2->getParent(),
                          TypeMatchKind::SameType, subFlags, trivial);
      }
      break;
    }

    case TypeKind::MetaType: {
      auto meta1 = cast<MetaTypeType>(desugar1);
      auto meta2 = cast<MetaTypeType>(desugar2);

      // metatype<B> < metatype<A> if A < B and both A and B are classes.
      TypeMatchKind subKind = TypeMatchKind::SameType;
      if (kind != TypeMatchKind::SameType &&
          (meta1->getInstanceType()->getClassOrBoundGenericClass() ||
           meta2->getInstanceType()->getClassOrBoundGenericClass()))
        subKind = std::min(kind, TypeMatchKind::Subtype);
      
      return matchTypes(meta1->getInstanceType(), meta2->getInstanceType(),
                        subKind, subFlags, trivial);
    }

    case TypeKind::Function: {
      auto func1 = cast<FunctionType>(desugar1);
      auto func2 = cast<FunctionType>(desugar2);
      return matchFunctionTypes(func1, func2, kind, flags, trivial);
    }

    case TypeKind::PolymorphicFunction:
      llvm_unreachable("Polymorphic function type should have been opened");

    case TypeKind::Array: {
      auto array1 = cast<ArrayType>(desugar1);
      auto array2 = cast<ArrayType>(desugar2);
      return matchTypes(array1->getBaseType(), array2->getBaseType(),
                        TypeMatchKind::SameType, subFlags, trivial);
    }

    case TypeKind::ProtocolComposition:
      // Existential types handled below.
      break;

    case TypeKind::LValue: {
      auto lvalue1 = cast<LValueType>(desugar1);
      auto lvalue2 = cast<LValueType>(desugar2);
      if (lvalue1->getQualifiers() != lvalue2->getQualifiers() &&
          !(kind >= TypeMatchKind::TrivialSubtype &&
            lvalue1->getQualifiers() < lvalue2->getQualifiers()))
        return SolutionKind::Error;

      return matchTypes(lvalue1->getObjectType(), lvalue2->getObjectType(),
                        TypeMatchKind::SameType, subFlags, trivial);
    }

    case TypeKind::UnboundGeneric:
      llvm_unreachable("Unbound generic type should have been opened");

    case TypeKind::BoundGenericClass:
    case TypeKind::BoundGenericOneOf:
    case TypeKind::BoundGenericStruct: {
      auto bound1 = cast<BoundGenericType>(desugar1);
      auto bound2 = cast<BoundGenericType>(desugar2);
      
      if (bound1->getDecl() == bound2->getDecl()) {
        // Match up the parents, exactly, if there are parents.
        SolutionKind result = SolutionKind::TriviallySolved;
        assert((bool)bound1->getParent() == (bool)bound2->getParent() &&
               "Mismatched parents of bound generics");
        if (bound1->getParent()) {
          switch (matchTypes(bound1->getParent(), bound2->getParent(),
                             TypeMatchKind::SameType, subFlags, trivial)) {
          case SolutionKind::Error:
            return SolutionKind::Error;

          case SolutionKind::TriviallySolved:
            break;

          case SolutionKind::Solved:
            result = SolutionKind::Solved;
            break;

          case SolutionKind::Unsolved:
            result = SolutionKind::Unsolved;
            break;
          }
        }

        // Match up the generic arguments, exactly.
        auto args1 = bound1->getGenericArgs();
        auto args2 = bound2->getGenericArgs();
        assert(args1.size() == args2.size() && "Mismatched generic args");
        for (unsigned i = 0, n = args1.size(); i != n; ++i) {
          switch (matchTypes(args1[i], args2[i], TypeMatchKind::SameType,
                             subFlags, trivial)) {
          case SolutionKind::Error:
            return SolutionKind::Error;

          case SolutionKind::TriviallySolved:
            break;

          case SolutionKind::Solved:
            result = SolutionKind::Solved;
            break;

          case SolutionKind::Unsolved:
            result = SolutionKind::Unsolved;
            break;
          }
        }

        return result;
      }
      break;
    }
    }
  }

  // FIXME: Materialization

  bool concrete = !typeVar1 && !typeVar2;
  if (concrete && kind >= TypeMatchKind::TrivialSubtype) {
    if (auto tuple2 = type2->getAs<TupleType>()) {
      // A scalar type is a trivial subtype of a one-element, non-variadic tuple
      // containing a single element if the scalar type is a subtype of
      // the type of that tuple's element.
      if (tuple2->getFields().size() == 1 &&
          !tuple2->getFields()[0].isVararg()) {
        return matchTypes(type1, tuple2->getElementType(0), kind, subFlags,
                          trivial);
      }

      // A scalar type can be converted to a tuple so long as there is at
      // most one non-defaulted element.
      if (kind >= TypeMatchKind::Conversion) {
        int scalarFieldIdx = tuple2->getFieldForScalarInit();
        if (scalarFieldIdx >= 0) {
          const auto &elt = tuple2->getFields()[scalarFieldIdx];
          auto scalarFieldTy = elt.isVararg()? elt.getVarargBaseTy()
                                             : elt.getType();
          return matchTypes(type1, scalarFieldTy, kind, subFlags, trivial);
        }
      }
    }

    // A class (or bound generic class) is a subtype of another class
    // (or bound generic class) if it is derived from that class.
    if (type1->getClassOrBoundGenericClass() &&
        type2->getClassOrBoundGenericClass()) {
      auto classDecl2 = type2->getClassOrBoundGenericClass();
      for (auto super1 = TC.getSuperClassOf(type1); super1;
           super1 = TC.getSuperClassOf(super1)) {
        if (super1->getClassOrBoundGenericClass() != classDecl2)
          continue;
        
        switch (auto result = matchTypes(super1, type2, TypeMatchKind::SameType,
                                         subFlags, trivial)) {
        case SolutionKind::Error:
          continue;

        case SolutionKind::Solved:
        case SolutionKind::TriviallySolved:
        case SolutionKind::Unsolved:
          return result;
        }
      }
    }
  }

  if (concrete && kind >= TypeMatchKind::Conversion) {
    // An lvalue of type T1 can be converted to a value of type T2 so long as
    // T1 is convertible to T2 (by loading the value).
    if (auto lvalue1 = type1->getAs<LValueType>()) {
      return matchTypes(lvalue1->getObjectType(), type2, kind, subFlags,
                        trivial);
    }

    // An expression can be converted to an auto-closure function type, creating
    // an implicit closure.
   if (auto function2 = type2->getAs<FunctionType>()) {
      if (function2->isAutoClosure()) {
        trivial = false;
        return matchTypes(type1, function2->getResult(), kind, subFlags,
                          trivial);
      }
    }
  }

  // For a subtyping relation involving two existential types, or a conversion
  // from any type, check whether the first type conforms to each of
  if (concrete &&
      (kind >= TypeMatchKind::Conversion ||
       (kind == TypeMatchKind::Subtype && type1->isExistentialType()))) {
    SmallVector<ProtocolDecl *, 4> protocols;
    if (!type1->hasTypeVariable() && type2->isExistentialType(protocols)) {
      for (auto proto : protocols) {
        if (!TC.conformsToProtocol(type1, proto))
          return SolutionKind::Error;
      }

      trivial = false;
      return SolutionKind::Solved;
    }
  }
  
  // A type can be constructed by passing an argument to one of its
  // constructors. This construction only applies to oneof and struct types
  // (or generic versions of oneof or struct types).
  if (kind == TypeMatchKind::Construction && isConstructibleType(type2)) {
    ConstructorLookup constructors(type2, TC.TU);
    if (constructors.isSuccess()) {
      auto &context = getASTContext();
      // FIXME: lame name
      auto name = context.getIdentifier("constructor");
      auto tv = createTypeVariable();

      // The constructor will have function type T -> T2, for a fresh type
      // variable T. Note that these constraints specifically require a
      // match on the result type because the constructors for oneofs and struct
      // types always return a value of exactly that type.
      addValueMemberConstraint(type2, name,
                               FunctionType::get(tv, type2, context));

      // The first type must be convertible to the constructor's argument type.
      addConstraint(ConstraintKind::Conversion, type1, tv);
      
      // FIXME: Do we want to consider conversion functions simultaneously with
      // constructors? Right now, we prefer constructors if they exist.
      return SolutionKind::Solved;
    }
  }


  // A nominal type can be converted to another type via a user-defined
  // conversion function.
  if (concrete && kind >= TypeMatchKind::Conversion &&
      type1->getNominalOrBoundGenericNominal()) {
    auto &context = getASTContext();
    auto name = context.getIdentifier("__conversion");
    MemberLookup &lookup = lookupMember(type1, name);
    if (lookup.isSuccess()) {
      auto inputTV = createTypeVariable();
      auto outputTV = createTypeVariable();

      // The conversion function will have function type TI -> TO, for fresh
      // type variables TI and TO.
      // FIXME: lame name!
      addValueMemberConstraint(type1, name,
                               FunctionType::get(inputTV, outputTV, context));

      // A conversion function must accept an empty parameter list ().
      addConstraint(ConstraintKind::Conversion, TupleType::getEmpty(context),
                    inputTV);

      // The output of the conversion function must be a subtype of the
      // type we're trying to convert to. The use of subtyping here eliminates
      // multiple-step user-defined conversions, which also eliminates concerns
      // about cyclic conversions causing infinite loops in the constraint
      // solver.
      addConstraint(ConstraintKind::Subtype, outputTV, type2);
      return SolutionKind::Solved;
    }
  }

  // If one of the types is a type variable, we leave this unsolved.
  if (typeVar1 || typeVar2)
    return SolutionKind::Unsolved;

  return SolutionKind::Error;
}
