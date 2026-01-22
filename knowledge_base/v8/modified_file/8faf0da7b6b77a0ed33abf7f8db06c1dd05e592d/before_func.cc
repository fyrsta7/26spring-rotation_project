void CodeGenerator::SmiOperation(Token::Value op,
                                 StaticType* type,
                                 Handle<Object> value,
                                 bool reversed,
                                 OverwriteMode overwrite_mode) {
  // NOTE: This is an attempt to inline (a bit) more of the code for
  // some possible smi operations (like + and -) when (at least) one
  // of the operands is a literal smi. With this optimization, the
  // performance of the system is increased by ~15%, and the generated
  // code size is increased by ~1% (measured on a combination of
  // different benchmarks).

  // TODO(199): Optimize some special cases of operations involving a
  // smi literal (multiply by 2, shift by 0, etc.).

  // Get the literal value.
  int int_value = Smi::cast(*value)->value();
  ASSERT(is_intn(int_value, kMaxSmiInlinedBits));

  switch (op) {
    case Token::ADD: {
      DeferredCode* deferred = NULL;
      if (!reversed) {
        deferred = new DeferredInlinedSmiAdd(this, int_value, overwrite_mode);
      } else {
        deferred = new DeferredInlinedSmiAddReversed(this, int_value,
                                                     overwrite_mode);
      }
      frame_->Pop(eax);
      __ add(Operand(eax), Immediate(value));
      __ j(overflow, deferred->enter(), not_taken);
      __ test(eax, Immediate(kSmiTagMask));
      __ j(not_zero, deferred->enter(), not_taken);
      __ bind(deferred->exit());
      frame_->Push(eax);
      break;
    }

    case Token::SUB: {
      DeferredCode* deferred = NULL;
      frame_->Pop(eax);
      if (!reversed) {
        deferred = new DeferredInlinedSmiSub(this, int_value, overwrite_mode);
        __ sub(Operand(eax), Immediate(value));
      } else {
        deferred = new DeferredInlinedSmiSubReversed(this, edx, overwrite_mode);
        __ mov(edx, Operand(eax));
        __ mov(eax, Immediate(value));
        __ sub(eax, Operand(edx));
      }
      __ j(overflow, deferred->enter(), not_taken);
      __ test(eax, Immediate(kSmiTagMask));
      __ j(not_zero, deferred->enter(), not_taken);
      __ bind(deferred->exit());
      frame_->Push(eax);
      break;
    }

    case Token::SAR: {
      if (reversed) {
        frame_->Pop(eax);
        frame_->Push(Immediate(value));
        frame_->Push(eax);
        GenericBinaryOperation(op, type, overwrite_mode);
      } else {
        int shift_value = int_value & 0x1f;  // only least significant 5 bits
        DeferredCode* deferred =
          new DeferredInlinedSmiOperation(this, Token::SAR, shift_value,
                                          overwrite_mode);
        frame_->Pop(eax);
        __ test(eax, Immediate(kSmiTagMask));
        __ j(not_zero, deferred->enter(), not_taken);
        __ sar(eax, shift_value);
        __ and_(eax, ~kSmiTagMask);
        __ bind(deferred->exit());
        frame_->Push(eax);
      }
      break;
    }

    case Token::SHR: {
      if (reversed) {
        frame_->Pop(eax);
        frame_->Push(Immediate(value));
        frame_->Push(eax);
        GenericBinaryOperation(op, type, overwrite_mode);
      } else {
        int shift_value = int_value & 0x1f;  // only least significant 5 bits
        DeferredCode* deferred =
        new DeferredInlinedSmiOperation(this, Token::SHR, shift_value,
                                        overwrite_mode);
        frame_->Pop(eax);
        __ test(eax, Immediate(kSmiTagMask));
        __ mov(ebx, Operand(eax));
        __ j(not_zero, deferred->enter(), not_taken);
        __ sar(ebx, kSmiTagSize);
        __ shr(ebx, shift_value);
        __ test(ebx, Immediate(0xc0000000));
        __ j(not_zero, deferred->enter(), not_taken);
        // tag result and store it in TOS (eax)
        ASSERT(kSmiTagSize == times_2);  // adjust code if not the case
        __ lea(eax, Operand(ebx, ebx, times_1, kSmiTag));
        __ bind(deferred->exit());
        frame_->Push(eax);
      }
      break;
    }

    case Token::SHL: {
      if (reversed) {
        frame_->Pop(eax);
        frame_->Push(Immediate(value));
        frame_->Push(eax);
        GenericBinaryOperation(op, type, overwrite_mode);
      } else {
        int shift_value = int_value & 0x1f;  // only least significant 5 bits
        DeferredCode* deferred =
        new DeferredInlinedSmiOperation(this, Token::SHL, shift_value,
                                        overwrite_mode);
        frame_->Pop(eax);
        __ test(eax, Immediate(kSmiTagMask));
        __ mov(ebx, Operand(eax));
        __ j(not_zero, deferred->enter(), not_taken);
        __ sar(ebx, kSmiTagSize);
        __ shl(ebx, shift_value);
        // Convert the int to a Smi, and check that it is in
        // the range of valid Smis.
        ASSERT(kSmiTagSize == times_2);  // Adjust code if not true.
        ASSERT(kSmiTag == 0);  // Adjust code if not true.
        __ add(ebx, Operand(ebx));
        __ j(overflow, deferred->enter(), not_taken);
        __ mov(eax, Operand(ebx));

        __ bind(deferred->exit());
        frame_->Push(eax);
      }
      break;
    }

    case Token::BIT_OR:
    case Token::BIT_XOR:
    case Token::BIT_AND: {
      DeferredCode* deferred = NULL;
      if (!reversed) {
        deferred =  new DeferredInlinedSmiOperation(this, op, int_value,
                                                    overwrite_mode);
      } else {
        deferred = new DeferredInlinedSmiOperationReversed(this, op, int_value,
                                                           overwrite_mode);
      }
      frame_->Pop(eax);
      __ test(eax, Immediate(kSmiTagMask));
      __ j(not_zero, deferred->enter(), not_taken);
      if (op == Token::BIT_AND) {
        __ and_(Operand(eax), Immediate(value));
      } else if (op == Token::BIT_XOR) {
        __ xor_(Operand(eax), Immediate(value));
      } else {
        ASSERT(op == Token::BIT_OR);
        __ or_(Operand(eax), Immediate(value));
      }
      __ bind(deferred->exit());
      frame_->Push(eax);
      break;
    }

    default: {
      if (!reversed) {
        frame_->Push(Immediate(value));
      } else {
        frame_->Pop(eax);
        frame_->Push(Immediate(value));
        frame_->Push(eax);
      }
      GenericBinaryOperation(op, type, overwrite_mode);
      break;
    }
  }
}
