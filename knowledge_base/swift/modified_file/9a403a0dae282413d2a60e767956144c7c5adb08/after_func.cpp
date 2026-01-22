MetadataResponse
swift::swift_getAssociatedTypeWitness(MetadataRequest request,
                                      WitnessTable *wtable,
                                      const Metadata *conformingType,
                                      const ProtocolRequirement *assocType) {
  const ProtocolConformanceDescriptor *conformance = wtable->Description;
  const ProtocolDescriptor *protocol = conformance->getProtocol();

  auto requirements = protocol->getRequirements();
  assert(assocType >= requirements.begin() &&
         assocType < requirements.end());
  const auto &req = *assocType;
  (void)req;
  assert(req.Flags.getKind() ==
           ProtocolRequirementFlags::Kind::AssociatedTypeAccessFunction);

  // If the low bit of the witness is clear, it's already a metadata pointer.
  unsigned witnessIndex = (assocType - requirements.begin()) +
    WitnessTableFirstRequirementOffset;
  auto witness = ((const void* const *)wtable)[witnessIndex];
  if (LLVM_LIKELY((uintptr_t(witness) &
         ProtocolRequirementFlags::AssociatedTypeMangledNameBit) == 0)) {
    // Cached metadata pointers are always complete.
    return MetadataResponse{(const Metadata *)witness, MetadataState::Complete};
  }

  // Find the mangled name.
  const char *mangledNameBase =
    (const char *)(uintptr_t(witness) &
                   ~ProtocolRequirementFlags::AssociatedTypeMangledNameBit);

  // Check whether the mangled name has the prefix byte indicating that
  // the mangled name is relative to the protocol itself.
  bool inProtocolContext = false;
  if ((uint8_t)*mangledNameBase ==
        ProtocolRequirementFlags::AssociatedTypeInProtocolContextByte) {
    inProtocolContext = true;
    ++mangledNameBase;
  }

  // Extract the mangled name itself.
  StringRef mangledName =
    Demangle::makeSymbolicMangledNameStringRef(mangledNameBase);

  // Demangle the associated type.
  const Metadata *assocTypeMetadata;
  if (inProtocolContext) {
    // The protocol's Self is the only generic parameter that can occur in the
    // type.
    assocTypeMetadata =
      _getTypeByMangledName(mangledName,
         [conformingType](unsigned depth, unsigned index) -> const Metadata * {
        if (depth == 0 && index == 0)
          return conformingType;

        return nullptr;
      });
  } else {
    // The generic parameters in the associated type name are those of the
    // conforming type.

    // For a class, chase the superclass chain up until we hit the
    // type that specified the conformance.
    auto originalConformingType = findConformingSuperclass(conformingType,
                                                           protocol);
    SubstGenericParametersFromMetadata substitutions(originalConformingType);
    assocTypeMetadata = _getTypeByMangledName(mangledName, substitutions);
  }

  if (!assocTypeMetadata) {
    auto conformingTypeNameInfo = swift_getTypeName(conformingType, true);
    StringRef conformingTypeName(conformingTypeNameInfo.data,
                                 conformingTypeNameInfo.length);
    StringRef assocTypeName = findAssociatedTypeName(protocol, assocType);
    fatalError(0,
               "failed to demangle witness for associated type '%s' in "
               "conformance '%s: %s' from mangled name '%s'\n",
               assocTypeName.str().c_str(),
               conformingTypeName.str().c_str(),
               protocol->Name.get(),
               mangledName.str().c_str());
  }


  assert((uintptr_t(assocTypeMetadata) &
            ProtocolRequirementFlags::AssociatedTypeMangledNameBit) == 0);

  // Check the metadata state.
  auto response = swift_checkMetadataState(request, assocTypeMetadata);

  // If the metadata was completed, record it in the witness table.
  if (response.State == MetadataState::Complete) {
    reinterpret_cast<const void**>(wtable)[witnessIndex] = assocTypeMetadata;
  }

  return response;
}
