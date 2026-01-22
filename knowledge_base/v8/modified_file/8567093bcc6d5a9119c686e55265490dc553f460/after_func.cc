Node* EffectControlLinearizer::LowerCheckSeqString(Node* node,
                                                   Node* frame_state) {
  Node* value = node->InputAt(0);

  Node* value_map = __ LoadField(AccessBuilder::ForMap(), value);
  Node* value_instance_type =
      __ LoadField(AccessBuilder::ForMapInstanceType(), value_map);

  Node* check = __ Word32Equal(
      __ Word32And(
          value_instance_type,
          __ Int32Constant(kStringRepresentationMask | kIsNotStringMask)),
      __ Int32Constant(kSeqStringTag | kStringTag));
  __ DeoptimizeIfNot(DeoptimizeReason::kWrongInstanceType, check, frame_state);
  return value;
}
