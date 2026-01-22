void SimdScalarLowering::LowerCompareOp(Node* node, SimdType input_rep_type,
                                        const Operator* op,
                                        bool invert_inputs) {
  DCHECK_EQ(2, node->InputCount());
  Node** rep_left = GetReplacementsWithType(node->InputAt(0), input_rep_type);
  Node** rep_right = GetReplacementsWithType(node->InputAt(1), input_rep_type);
  int num_lanes = NumLanes(input_rep_type);
  Node** rep_node = zone()->NewArray<Node*>(num_lanes);
  for (int i = 0; i < num_lanes; ++i) {
    Node* cmp_result = nullptr;
    if (invert_inputs) {
      cmp_result = graph()->NewNode(op, rep_right[i], rep_left[i]);
    } else {
      cmp_result = graph()->NewNode(op, rep_left[i], rep_right[i]);
    }
    Diamond d_cmp(graph(), common(),
                  graph()->NewNode(machine()->Word32Equal(), cmp_result,
                                   mcgraph_->Int32Constant(0)));
    MachineRepresentation rep =
        (input_rep_type == SimdType::kFloat32x4)
            ? MachineRepresentation::kWord32
            : MachineTypeFrom(input_rep_type).representation();
    rep_node[i] =
        d_cmp.Phi(rep, mcgraph_->Int32Constant(0), mcgraph_->Int32Constant(-1));
  }
  ReplaceNode(node, rep_node, num_lanes);
}
