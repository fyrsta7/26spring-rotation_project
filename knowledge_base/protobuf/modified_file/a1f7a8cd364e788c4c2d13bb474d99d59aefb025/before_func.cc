bool DescriptorPool::IsSubSymbolOfBuiltType(absl::string_view name) const {
  auto prefix = std::string(name);
  for (;;) {
    std::string::size_type dot_pos = prefix.find_last_of('.');
    if (dot_pos == std::string::npos) {
      break;
    }
    prefix = prefix.substr(0, dot_pos);
    Symbol symbol = tables_->FindSymbol(prefix);
    // If the symbol type is anything other than PACKAGE, then its complete
    // definition is already known.
    if (!symbol.IsNull() && !symbol.IsPackage()) {
      return true;
    }
  }
  if (underlay_ != nullptr) {
    // Check to see if any prefix of this symbol exists in the underlay.
    return underlay_->IsSubSymbolOfBuiltType(name);
  }
  return false;
}
