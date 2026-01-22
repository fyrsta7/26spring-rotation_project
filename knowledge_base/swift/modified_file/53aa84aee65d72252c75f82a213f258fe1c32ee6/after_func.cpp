bool swift::isWhitelistedSpecialization(StringRef SpecName) {
  // The whitelist of classes and functions from the stdlib,
  // whose specializations we want to preserve.
  ArrayRef<StringRef> Whitelist = {
      "Array",
      "_ArrayBuffer",
      "_ContiguousArrayBuffer",
      "Range",
      "RangeGenerator",
      "_allocateUninitializedArray",
      "UTF8",
      "UTF16",
      "String",
      "_StringBuffer",
      "_toStringReadOnlyPrintable",
  };

  // TODO: Once there is an efficient API to check if
  // a given symbol is a specialization of a specific type,
  // use it instead. Doing demangling just for this check
  // is just wasteful.
  auto DemangledNameString =
     swift::Demangle::demangleSymbolAsString(SpecName);

  StringRef DemangledName = DemangledNameString;

  auto pos = DemangledName.find("generic ", 0);
  if (pos == StringRef::npos)
    return false;

  // Create "of Swift"
  llvm::SmallString<64> OfString;
  llvm::raw_svector_ostream buffer(OfString);
  buffer << "of ";
  buffer << STDLIB_NAME <<'.';

  StringRef OfStr = buffer.str();

  pos = DemangledName.find(OfStr, pos);

  if (pos == StringRef::npos)
    return false;

  pos += OfStr.size();

  for(auto Name: Whitelist) {
    auto pos1 = DemangledName.find(Name, pos);
    if (pos1 == pos && !isalpha(DemangledName[pos1+Name.size()])) {
      return true;
    }
  }

  return false;
}
