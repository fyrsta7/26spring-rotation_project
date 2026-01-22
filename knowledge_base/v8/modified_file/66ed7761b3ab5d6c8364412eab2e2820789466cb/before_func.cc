  void UnaryOpWithFeedback() {
    VARIABLE(var_value, MachineRepresentation::kTagged, GetAccumulator());
    Node* slot_index = BytecodeOperandIdx(0);
    Node* maybe_feedback_vector = LoadFeedbackVector();

    VARIABLE(var_result, MachineRepresentation::kTagged);
    VARIABLE(var_float_value, MachineRepresentation::kFloat64);
    TVARIABLE(Smi, var_feedback, SmiConstant(BinaryOperationFeedback::kNone));
    Variable* loop_vars[] = {&var_value, &var_feedback};
    Label start(this, arraysize(loop_vars), loop_vars), end(this);
    Label do_float_op(this, &var_float_value);
    Goto(&start);
    // We might have to try again after ToNumeric conversion.
    BIND(&start);
    {
      Label if_smi(this), if_heapnumber(this), if_bigint(this);
      Label if_oddball(this), if_other(this);
      Node* value = var_value.value();
      GotoIf(TaggedIsSmi(value), &if_smi);
      Node* map = LoadMap(value);
      GotoIf(IsHeapNumberMap(map), &if_heapnumber);
      Node* instance_type = LoadMapInstanceType(map);
      GotoIf(IsBigIntInstanceType(instance_type), &if_bigint);
      Branch(InstanceTypeEqual(instance_type, ODDBALL_TYPE), &if_oddball,
             &if_other);

      BIND(&if_smi);
      {
        var_result.Bind(
            SmiOp(CAST(value), &var_feedback, &do_float_op, &var_float_value));
        Goto(&end);
      }

      BIND(&if_heapnumber);
      {
        var_float_value.Bind(LoadHeapNumberValue(value));
        Goto(&do_float_op);
      }

      BIND(&if_bigint);
      {
        var_result.Bind(BigIntOp(value));
        CombineFeedback(&var_feedback, BinaryOperationFeedback::kBigInt);
        Goto(&end);
      }

      BIND(&if_oddball);
      {
        // We do not require an Or with earlier feedback here because once we
        // convert the value to a number, we cannot reach this path. We can
        // only reach this path on the first pass when the feedback is kNone.
        CSA_ASSERT(this, SmiEqual(var_feedback.value(),
                                  SmiConstant(BinaryOperationFeedback::kNone)));
        OverwriteFeedback(&var_feedback,
                          BinaryOperationFeedback::kNumberOrOddball);
        var_value.Bind(LoadObjectField(value, Oddball::kToNumberOffset));
        Goto(&start);
      }

      BIND(&if_other);
      {
        // We do not require an Or with earlier feedback here because once we
        // convert the value to a number, we cannot reach this path. We can
        // only reach this path on the first pass when the feedback is kNone.
        CSA_ASSERT(this, SmiEqual(var_feedback.value(),
                                  SmiConstant(BinaryOperationFeedback::kNone)));
        OverwriteFeedback(&var_feedback, BinaryOperationFeedback::kAny);
        var_value.Bind(
            CallBuiltin(Builtins::kNonNumberToNumeric, GetContext(), value));
        Goto(&start);
      }
    }

    BIND(&do_float_op);
    {
      CombineFeedback(&var_feedback, BinaryOperationFeedback::kNumber);
      var_result.Bind(
          AllocateHeapNumberWithValue(FloatOp(var_float_value.value())));
      Goto(&end);
    }

    BIND(&end);
    UpdateFeedback(var_feedback.value(), maybe_feedback_vector, slot_index);
    SetAccumulator(var_result.value());
    Dispatch();
  }
