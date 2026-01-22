HControlInstruction* HOptimizedGraphBuilder::BuildCompareInstruction(
    Token::Value op,
    HValue* left,
    HValue* right,
    Handle<Type> left_type,
    Handle<Type> right_type,
    Handle<Type> combined_type,
    int left_position,
    int right_position,
    BailoutId bailout_id) {
  Representation left_rep = Representation::FromType(left_type);
  Representation right_rep = Representation::FromType(right_type);
  Representation combined_rep = Representation::FromType(combined_type);

  if (combined_type->Is(Type::Receiver())) {
    if (Token::IsEqualityOp(op)) {
      // Can we get away with map check and not instance type check?
      if (combined_type->IsClass()) {
        Handle<Map> map = combined_type->AsClass();
        AddCheckMap(left, map);
        AddCheckMap(right, map);
        HCompareObjectEqAndBranch* result =
            New<HCompareObjectEqAndBranch>(left, right);
        if (FLAG_emit_opt_code_positions) {
          result->set_operand_position(zone(), 0, left_position);
          result->set_operand_position(zone(), 1, right_position);
        }
        return result;
      } else {
        BuildCheckHeapObject(left);
        Add<HCheckInstanceType>(left, HCheckInstanceType::IS_SPEC_OBJECT);
        BuildCheckHeapObject(right);
        Add<HCheckInstanceType>(right, HCheckInstanceType::IS_SPEC_OBJECT);
        HCompareObjectEqAndBranch* result =
            New<HCompareObjectEqAndBranch>(left, right);
        return result;
      }
    } else {
      Bailout(kUnsupportedNonPrimitiveCompare);
      return NULL;
    }
  } else if (combined_type->Is(Type::InternalizedString()) &&
             Token::IsEqualityOp(op)) {
    BuildCheckHeapObject(left);
    Add<HCheckInstanceType>(left, HCheckInstanceType::IS_INTERNALIZED_STRING);
    BuildCheckHeapObject(right);
    Add<HCheckInstanceType>(right, HCheckInstanceType::IS_INTERNALIZED_STRING);
    HCompareObjectEqAndBranch* result =
        New<HCompareObjectEqAndBranch>(left, right);
    return result;
  } else if (combined_type->Is(Type::String())) {
    BuildCheckHeapObject(left);
    Add<HCheckInstanceType>(left, HCheckInstanceType::IS_STRING);
    BuildCheckHeapObject(right);
    Add<HCheckInstanceType>(right, HCheckInstanceType::IS_STRING);
    HStringCompareAndBranch* result =
        New<HStringCompareAndBranch>(left, right, op);
    return result;
  } else {
    if (combined_rep.IsTagged() || combined_rep.IsNone()) {
      HCompareGeneric* result = Add<HCompareGeneric>(left, right, op);
      result->set_observed_input_representation(1, left_rep);
      result->set_observed_input_representation(2, right_rep);
      if (result->HasObservableSideEffects()) {
        Push(result);
        AddSimulate(bailout_id, REMOVABLE_SIMULATE);
        Drop(1);
      }
      // TODO(jkummerow): Can we make this more efficient?
      HBranch* branch = New<HBranch>(result);
      return branch;
    } else {
      HCompareNumericAndBranch* result =
          New<HCompareNumericAndBranch>(left, right, op);
      result->set_observed_input_representation(left_rep, right_rep);
      if (FLAG_emit_opt_code_positions) {
        result->SetOperandPositions(zone(), left_position, right_position);
      }
      return result;
    }
  }
}
