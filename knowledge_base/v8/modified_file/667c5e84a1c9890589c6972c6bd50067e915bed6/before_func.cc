void CodeGenerator::VisitCompareOperation(CompareOperation* node) {
  Comment cmnt(masm_, "[ CompareOperation");

  // Get the expressions from the node.
  Expression* left = node->left();
  Expression* right = node->right();
  Token::Value op = node->op();
  // To make typeof testing for natives implemented in JavaScript really
  // efficient, we generate special code for expressions of the form:
  // 'typeof <expression> == <string>'.
  UnaryOperation* operation = left->AsUnaryOperation();
  if ((op == Token::EQ || op == Token::EQ_STRICT) &&
      (operation != NULL && operation->op() == Token::TYPEOF) &&
      (right->AsLiteral() != NULL &&
       right->AsLiteral()->handle()->IsString())) {
    Handle<String> check(Handle<String>::cast(right->AsLiteral()->handle()));

    // Load the operand and move it to a register.
    LoadTypeofExpression(operation->expression());
    Result answer = frame_->Pop();
    answer.ToRegister();

    if (check->Equals(Heap::number_symbol())) {
      Condition is_smi = masm_->CheckSmi(answer.reg());
      destination()->true_target()->Branch(is_smi);
      frame_->Spill(answer.reg());
      __ movq(answer.reg(), FieldOperand(answer.reg(), HeapObject::kMapOffset));
      __ CompareRoot(answer.reg(), Heap::kHeapNumberMapRootIndex);
      answer.Unuse();
      destination()->Split(equal);

    } else if (check->Equals(Heap::string_symbol())) {
      Condition is_smi = masm_->CheckSmi(answer.reg());
      destination()->false_target()->Branch(is_smi);

      // It can be an undetectable string object.
      __ movq(kScratchRegister,
              FieldOperand(answer.reg(), HeapObject::kMapOffset));
      __ testb(FieldOperand(kScratchRegister, Map::kBitFieldOffset),
               Immediate(1 << Map::kIsUndetectable));
      destination()->false_target()->Branch(not_zero);
      __ CmpInstanceType(kScratchRegister, FIRST_NONSTRING_TYPE);
      answer.Unuse();
      destination()->Split(below);  // Unsigned byte comparison needed.

    } else if (check->Equals(Heap::boolean_symbol())) {
      __ CompareRoot(answer.reg(), Heap::kTrueValueRootIndex);
      destination()->true_target()->Branch(equal);
      __ CompareRoot(answer.reg(), Heap::kFalseValueRootIndex);
      answer.Unuse();
      destination()->Split(equal);

    } else if (check->Equals(Heap::undefined_symbol())) {
      __ CompareRoot(answer.reg(), Heap::kUndefinedValueRootIndex);
      destination()->true_target()->Branch(equal);

      Condition is_smi = masm_->CheckSmi(answer.reg());
      destination()->false_target()->Branch(is_smi);

      // It can be an undetectable object.
      __ movq(kScratchRegister,
              FieldOperand(answer.reg(), HeapObject::kMapOffset));
      __ testb(FieldOperand(kScratchRegister, Map::kBitFieldOffset),
               Immediate(1 << Map::kIsUndetectable));
      answer.Unuse();
      destination()->Split(not_zero);

    } else if (check->Equals(Heap::function_symbol())) {
      Condition is_smi = masm_->CheckSmi(answer.reg());
      destination()->false_target()->Branch(is_smi);
      frame_->Spill(answer.reg());
      __ CmpObjectType(answer.reg(), JS_FUNCTION_TYPE, answer.reg());
      destination()->true_target()->Branch(equal);
      // Regular expressions are callable so typeof == 'function'.
      __ CmpInstanceType(answer.reg(), JS_REGEXP_TYPE);
      answer.Unuse();
      destination()->Split(equal);

    } else if (check->Equals(Heap::object_symbol())) {
      Condition is_smi = masm_->CheckSmi(answer.reg());
      destination()->false_target()->Branch(is_smi);
      __ CompareRoot(answer.reg(), Heap::kNullValueRootIndex);
      destination()->true_target()->Branch(equal);

      // Regular expressions are typeof == 'function', not 'object'.
      __ CmpObjectType(answer.reg(), JS_REGEXP_TYPE, kScratchRegister);
      destination()->false_target()->Branch(equal);

      // It can be an undetectable object.
      __ testb(FieldOperand(kScratchRegister, Map::kBitFieldOffset),
               Immediate(1 << Map::kIsUndetectable));
      destination()->false_target()->Branch(not_zero);
      __ CmpInstanceType(kScratchRegister, FIRST_JS_OBJECT_TYPE);
      destination()->false_target()->Branch(below);
      __ CmpInstanceType(kScratchRegister, LAST_JS_OBJECT_TYPE);
      answer.Unuse();
      destination()->Split(below_equal);
    } else {
      // Uncommon case: typeof testing against a string literal that is
      // never returned from the typeof operator.
      answer.Unuse();
      destination()->Goto(false);
    }
    return;
  }

  Condition cc = no_condition;
  bool strict = false;
  switch (op) {
    case Token::EQ_STRICT:
      strict = true;
      // Fall through
    case Token::EQ:
      cc = equal;
      break;
    case Token::LT:
      cc = less;
      break;
    case Token::GT:
      cc = greater;
      break;
    case Token::LTE:
      cc = less_equal;
      break;
    case Token::GTE:
      cc = greater_equal;
      break;
    case Token::IN: {
      Load(left);
      Load(right);
      Result answer = frame_->InvokeBuiltin(Builtins::IN, CALL_FUNCTION, 2);
      frame_->Push(&answer);  // push the result
      return;
    }
    case Token::INSTANCEOF: {
      Load(left);
      Load(right);
      InstanceofStub stub;
      Result answer = frame_->CallStub(&stub, 2);
      answer.ToRegister();
      __ testq(answer.reg(), answer.reg());
      answer.Unuse();
      destination()->Split(zero);
      return;
    }
    default:
      UNREACHABLE();
  }
  Load(left);
  Load(right);
  Comparison(node, cc, strict, destination());
}
