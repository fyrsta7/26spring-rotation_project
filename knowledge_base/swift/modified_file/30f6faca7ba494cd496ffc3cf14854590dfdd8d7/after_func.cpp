    Type getFixedType(TypeVariableType *typeVar, bool isRepresentative = false){
      if (!isRepresentative)
        typeVar = getRepresentative(typeVar);
      auto known = FixedTypes.find(typeVar);
      if (known == FixedTypes.end()) {
        if (Parent)
          return Parent->getFixedType(typeVar, /*isRepresentative=*/true);

        return Type();
      }

      return known->second;
    }
