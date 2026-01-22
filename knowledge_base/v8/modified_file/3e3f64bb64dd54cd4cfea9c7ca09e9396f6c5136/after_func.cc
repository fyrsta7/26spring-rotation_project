void MaglevGraphBuilder::VisitTypeOf() {
  ValueNode* value = GetAccumulatorTagged();
  // TODO(victorgomes): Add a JSFunction type to Maglev.
  if (CheckType(value, NodeType::kBoolean)) {
    SetAccumulator(GetRootConstant(RootIndex::kboolean_string));
  } else if (CheckType(value, NodeType::kNumber)) {
    SetAccumulator(GetRootConstant(RootIndex::knumber_string));
  } else if (CheckType(value, NodeType::kString)) {
    SetAccumulator(GetRootConstant(RootIndex::kstring_string));
  } else if (CheckType(value, NodeType::kSymbol)) {
    SetAccumulator(GetRootConstant(RootIndex::ksymbol_string));
  } else if (IsUndefinedValue(value)) {
    SetAccumulator(GetRootConstant(RootIndex::kundefined_string));
  } else {
    SetAccumulator(BuildCallBuiltin<Builtin::kTypeof>({value}));
  }
}
