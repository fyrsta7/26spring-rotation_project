/// Emit a boolean expression as a control-flow condition.
///
/// \param hasFalseCode - true if the false branch doesn't just lead
/// to the fallthrough.
/// \param invertValue - true if this routine should invert the value before
/// testing true/false.
Condition IRGenFunction::emitCondition(Expr *E, bool hasFalseCode,
                                       bool invertValue) {
  assert(Builder.hasValidIP() && "emitting condition at unreachable point");

  // Sema forces conditions to have Builtin.i1 type, which guarantees this.
  // TODO: special-case interesting condition expressions.
  llvm::Value *V;
  {
    FullExpr Scope(*this);
    V = emitAsPrimitiveScalar(E);
  }
  assert(V->getType()->isIntegerTy(1));

  llvm::BasicBlock *trueBB, *falseBB, *contBB;

  // Check for a constant condition.
  if (llvm::ConstantInt *C = dyn_cast<llvm::ConstantInt>(V)) {
    // If the condition is constant false, ignore the true branch.  We
    // will fall into the false branch unless there is none.
    if (C->isZero() == !invertValue) {
      trueBB = nullptr;
      falseBB = (hasFalseCode ? Builder.GetInsertBlock() : nullptr);

    // Otherwise, ignore the false branch.  We will fall into the true
    // branch.
    } else {
      trueBB = Builder.GetInsertBlock();
      falseBB = nullptr;
    }

    // There is no separate continuation block.
    contBB = nullptr;

  // Otherwise, the condition requires a conditional branch.
  } else {
    // If requested, invert the value.
    if (invertValue)
      V = Builder.CreateXor(V,
                            llvm::Constant::getIntegerValue(IGM.Int1Ty,
                                                            llvm::APInt(1, 1)));
    
    contBB = createBasicBlock("condition.cont");
    trueBB = createBasicBlock("if.true");

    llvm::BasicBlock *falseDestBB;
    if (hasFalseCode) {
      falseBB = falseDestBB = createBasicBlock("if.false");
    } else {
      falseBB = nullptr;
      falseDestBB = contBB;
    }

    Builder.CreateCondBr(V, trueBB, falseDestBB);
  }

  return Condition(trueBB, falseBB, contBB);
}