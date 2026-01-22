void FastCodeGenerator::VisitCountOperation(CountOperation* expr) {
  Comment cmnt(masm_, "[ CountOperation");

  // Expression can only be a property, a global or a (parameter or local)
  // slot. Variables with rewrite to .arguments are treated as KEYED_PROPERTY.
  enum LhsKind { VARIABLE, NAMED_PROPERTY, KEYED_PROPERTY };
  LhsKind assign_type = VARIABLE;
  Property* prop = expr->expression()->AsProperty();
  // In case of a property we use the uninitialized expression context
  // of the key to detect a named property.
  if (prop != NULL) {
    assign_type = (prop->key()->context() == Expression::kUninitialized)
        ? NAMED_PROPERTY
        : KEYED_PROPERTY;
  }

  // Evaluate expression and get value.
  if (assign_type == VARIABLE) {
    ASSERT(expr->expression()->AsVariableProxy()->var() != NULL);
    EmitVariableLoad(expr->expression()->AsVariableProxy()->var(),
                     Expression::kValue);
  } else {
    // Reserve space for result of postfix operation.
    if (expr->is_postfix() && expr->context() != Expression::kEffect) {
      ASSERT(expr->context() != Expression::kUninitialized);
      __ mov(ip, Operand(Smi::FromInt(0)));
      __ push(ip);
    }
    Visit(prop->obj());
    ASSERT_EQ(Expression::kValue, prop->obj()->context());
    if (assign_type == NAMED_PROPERTY) {
      EmitNamedPropertyLoad(prop, Expression::kValue);
    } else {
      Visit(prop->key());
      ASSERT_EQ(Expression::kValue, prop->key()->context());
      EmitKeyedPropertyLoad(prop, Expression::kValue);
    }
  }

  // Convert to number.
  __ InvokeBuiltin(Builtins::TO_NUMBER, CALL_JS);

  // Save result for postfix expressions.
  if (expr->is_postfix()) {
    switch (expr->context()) {
      case Expression::kUninitialized:
        UNREACHABLE();
      case Expression::kEffect:
        // Do not save result.
        break;
      case Expression::kValue:  // Fall through
      case Expression::kTest:  // Fall through
      case Expression::kTestValue:  // Fall through
      case Expression::kValueTest:
        // Save the result on the stack. If we have a named or keyed property
        // we store the result under the receiver that is currently on top
        // of the stack.
        switch (assign_type) {
          case VARIABLE:
            __ push(r0);
            break;
          case NAMED_PROPERTY:
            __ str(r0, MemOperand(sp, kPointerSize));
            break;
          case KEYED_PROPERTY:
            __ str(r0, MemOperand(sp, 2 * kPointerSize));
            break;
        }
        break;
    }
  }

  // Call runtime for +1/-1.
  if (expr->op() == Token::INC) {
    __ mov(ip, Operand(Smi::FromInt(1)));
  } else {
    __ mov(ip, Operand(Smi::FromInt(-1)));
  }
  __ stm(db_w, sp, ip.bit() | r0.bit());
  __ CallRuntime(Runtime::kNumberAdd, 2);

  // Store the value returned in r0.
  switch (assign_type) {
    case VARIABLE:
      __ push(r0);
      if (expr->is_postfix()) {
        EmitVariableAssignment(expr->expression()->AsVariableProxy()->var(),
                               Expression::kEffect);
        // For all contexts except kEffect: We have the result on
        // top of the stack.
        if (expr->context() != Expression::kEffect) {
          MoveTOS(expr->context());
        }
      } else {
        EmitVariableAssignment(expr->expression()->AsVariableProxy()->var(),
                               expr->context());
      }
      break;
    case NAMED_PROPERTY: {
      __ mov(r2, Operand(prop->key()->AsLiteral()->handle()));
      Handle<Code> ic(Builtins::builtin(Builtins::StoreIC_Initialize));
      __ Call(ic, RelocInfo::CODE_TARGET);
      if (expr->is_postfix()) {
        __ Drop(1);  // Result is on the stack under the receiver.
        if (expr->context() != Expression::kEffect) {
          MoveTOS(expr->context());
        }
      } else {
        DropAndMove(expr->context(), r0);
      }
      break;
    }
    case KEYED_PROPERTY: {
      Handle<Code> ic(Builtins::builtin(Builtins::KeyedStoreIC_Initialize));
      __ Call(ic, RelocInfo::CODE_TARGET);
      if (expr->is_postfix()) {
        __ Drop(2);  // Result is on the stack under the key and the receiver.
        if (expr->context() != Expression::kEffect) {
          MoveTOS(expr->context());
        }
      } else {
        DropAndMove(expr->context(), r0, 2);
      }
      break;
    }
  }
}
