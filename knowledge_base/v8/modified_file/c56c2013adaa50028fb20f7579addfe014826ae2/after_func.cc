Node* EffectControlLinearizer::LowerCheckedUint32ToInt32(Node* node,
                                                         Node* frame_state) {
  Node* value = node->InputAt(0);
  Node* unsafe = __ Int32LessThan(value, __ Int32Constant(0));
  __ DeoptimizeIf(DeoptimizeReason::kLostPrecision, unsafe, frame_state);
  return value;
}
