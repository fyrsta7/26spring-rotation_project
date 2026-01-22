Node* EffectControlLinearizer::LowerCheckedUint32ToInt32(Node* node,
                                                         Node* frame_state) {
  Node* value = node->InputAt(0);
  Node* max_int = __ Int32Constant(std::numeric_limits<int32_t>::max());
  Node* is_safe = __ Uint32LessThanOrEqual(value, max_int);
  __ DeoptimizeUnless(DeoptimizeReason::kLostPrecision, is_safe, frame_state);
  return value;
}
