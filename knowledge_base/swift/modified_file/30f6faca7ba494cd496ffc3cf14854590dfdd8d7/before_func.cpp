    Type getFixedType(TypeVariableType *typeVar) {
      typeVar = getRepresentative(typeVar);
      auto known = FixedTypes.find(typeVar);
      if (known == FixedTypes.end()) {
        if (Parent)
          return Parent->getFixedType(typeVar);

        return Type();
      }

      return known->second;
    }
