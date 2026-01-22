                global_for_const);
  }

  return Status::OK();
}

Status IrEmitter::HandleConstant(HloInstruction* constant) {
  VLOG(2) << "HandleConstant: " << constant->ToString();
  // IrEmitter::EmitConstantGlobals has already taken care of emitting the body
  // of the constant.
  return EmitTargetAddressForOp(constant);
}

Status IrEmitter::HandleCopy(HloInstruction* copy) {
  if (copy->shape().IsTuple() ||
      (copy->shape().IsArray() &&
