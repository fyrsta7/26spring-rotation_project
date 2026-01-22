bool DescriptorPool::IsSubSymbolOfBuiltType(absl::string_view name) const {
  for (size_t pos = name.find('.'); pos != name.npos;
       pos = name.find('.', pos + 1)) {
    auto prefix = name.substr(0, pos);
    Symbol symbol = tables_->FindSymbol(prefix);
    if (symbol.IsNull()) {
      break;
    }
    if (!symbol.IsPackage()) {
      // If the symbol type is anything other than PACKAGE, then its complete
      // definition is already known.
      return true;
    }
  }
  if (underlay_ != nullptr) {
    // Check to see if any prefix of this symbol exists in the underlay.
    return underlay_->IsSubSymbolOfBuiltType(name);
  }
  return false;
}
