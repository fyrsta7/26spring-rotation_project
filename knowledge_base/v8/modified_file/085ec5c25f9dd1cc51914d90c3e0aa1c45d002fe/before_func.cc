Node* SimplifiedLowering::Int32Abs(Node* const node) {
  Node* const zero = jsgraph()->Int32Constant(0);
  Node* const input = node->InputAt(0);

  // if 0 < input then input else 0 - input
  return graph()->NewNode(
      common()->Select(MachineRepresentation::kWord32, BranchHint::kTrue),
      graph()->NewNode(machine()->Int32LessThan(), zero, input), input,
      graph()->NewNode(machine()->Int32Sub(), zero, input));
}
