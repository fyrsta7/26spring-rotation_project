Node* EffectControlLinearizer::LowerCheckSeqString(Node* node,
                                                   Node* frame_state) {
  Node* value = node->InputAt(0);

  Node* value_map = __ LoadField(AccessBuilder::ForMap(), value);
  Node* value_instance_type =
      __ LoadField(AccessBuilder::ForMapInstanceType(), value_map);

  Node* is_string = __ Uint32LessThan(value_instance_type,
                                      __ Uint32Constant(FIRST_NONSTRING_TYPE));
  Node* is_sequential =
      __ Word32Equal(__ Word32And(value_instance_type,
                                  __ Int32Constant(kStringRepresentationMask)),
                     __ Int32Constant(kSeqStringTag));
  Node* is_sequential_string = __ Word32And(is_string, is_sequential);

  __ DeoptimizeIfNot(DeoptimizeReason::kWrongInstanceType, is_sequential_string,
                     frame_state);
  return value;
}
