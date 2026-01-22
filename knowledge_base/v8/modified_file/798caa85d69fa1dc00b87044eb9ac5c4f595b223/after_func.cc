void InstructionSelector::VisitWord64ReverseBytes(Node* node) {
  S390OperandGenerator g(this);
  NodeMatcher input(node->InputAt(0));
  if (CanCover(node, input.node()) && input.IsLoad()) {
    LoadRepresentation load_rep = LoadRepresentationOf(input.node()->op());
    if (load_rep.representation() == MachineRepresentation::kWord64) {
      Node* base = input.node()->InputAt(0);
      Node* offset = input.node()->InputAt(1);
      Emit(kS390_LoadReverse64 | AddressingModeField::encode(kMode_MRR),
           // TODO(miladfarca): one of the base and offset can be imm.
           g.DefineAsRegister(node), g.UseRegister(base),
           g.UseRegister(offset));
      return;
    }
  }
  Emit(kS390_LoadReverse64RR, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)));
}
