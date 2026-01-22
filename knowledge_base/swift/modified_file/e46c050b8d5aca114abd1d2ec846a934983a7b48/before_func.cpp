    }

    /// Emit the source value for parameters.
    llvm::Value *emitSourceForParameters(const Source &source,
                                         Explosion &in,
                                         const GetParameterFn &getParameter) {
      switch (source.getKind()) {
      case SourceKind::Metadata:
        return getParameter(source.getParamIndex());

      case SourceKind::ClassPointer: {
        unsigned paramIndex = source.getParamIndex();
        llvm::Value *instanceRef = getParameter(paramIndex);
        SILType instanceType =
          SILType::getPrimitiveObjectType(getArgTypeInContext(paramIndex));
        return emitDynamicTypeOfHeapObject(IGF, instanceRef, instanceType);
      }

      case SourceKind::GenericLValueMetadata: {
        llvm::Value *metatype = in.claimNext();
        metatype->setName("Self");

        // Mark this as the cached metatype for the l-value's object type.
        CanType argTy = getArgTypeInContext(source.getParamIndex());
        IGF.setUnscopedLocalTypeData(argTy, LocalTypeData::Metatype, metatype);
        return metatype;
      }

      case SourceKind::WitnessSelf:
      case SourceKind::WitnessExtraData: {
        // The 'Self' parameter is provided last.
        // TODO: For default implementations, the witness table pointer for
        // the 'Self : P' conformance must be provided last along with the
        // metatype.
        llvm::Value *metatype = in.takeLast();
        metatype->setName("Self");
        return metatype;
      }
