LogicalResult MoveIntoAssumingOpMatchAndRewrite(Operation *op,
                                                PatternRewriter &rewriter) {
  // Find a preceding `assuming` op with nothing but side effect-free operations
  // in between.
  Operation *prev = op->getPrevNode();
  while (prev != nullptr && !llvm::isa<shape::AssumingOp>(prev) &&
         IsMovable(prev)) {
    prev = prev->getPrevNode();
  }
  auto assuming_op = llvm::dyn_cast_or_null<shape::AssumingOp>(prev);
  if (!assuming_op) return failure();

  // Make sure that all operands will be available after moving.
  auto is_available = [&](Value v) {
    Operation *def = v.getDefiningOp();
    return def == nullptr || (def->getBlock() == op->getBlock() &&
                              !assuming_op->isBeforeInBlock(def));
  };
  if (!llvm::all_of(op->getOperands(), is_available)) return failure();

  Block *body = assuming_op.getBody();
  auto yield_op = cast<shape::AssumingYieldOp>(body->getTerminator());

  // Find the operands to use if the op was within the assuming region. We
  // will later use their copies, as we copy the assuming op and its body.
  SmallVector<Value, 8> new_operands_unmapped =
      llvm::to_vector<8>(llvm::map_range(op->getOperands(), [&](Value v) {
        for (const auto &result : llvm::enumerate(assuming_op->getResults())) {
          if (result.value() == v) return yield_op->getOperand(result.index());
        }
        return v;
      }));

  // Insert the rewritten assuming op right before the old one.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(assuming_op);
  auto new_assuming_op = rewriter.create<shape::AssumingOp>(
      assuming_op.getLoc(), assuming_op.getWitness(),
      [&](OpBuilder &b, Location) {
        // Copy body.
        BlockAndValueMapping mapping;
        for (auto &nested : body->without_terminator())
          b.clone(nested, mapping);

        // Copy op into the new body and use the mapped operands.
        for (auto it : llvm::zip(op->getOperands(), new_operands_unmapped)) {
          Value old_operand, new_operand_unmapped;
          std::tie(old_operand, new_operand_unmapped) = it;
          mapping.map(old_operand,
                      mapping.lookupOrDefault(new_operand_unmapped));
        }
        Operation *new_op = b.clone(*op, mapping);

        // Yield the previous results and also the new ones.
        auto mapped_results = llvm::to_vector<8>(llvm::map_range(
            yield_op.getOperands(),
            [&](Value v) { return mapping.lookupOrDefault(v); }));
        mapped_results.append(new_op->getResults().begin(),
                              new_op->getResults().end());
        return mapped_results;
      });

  // Replace the assuming op and the root op with the corresponding result
  // value.
  ValueRange new_assuming_op_results = new_assuming_op->getResults();
  rewriter.replaceOp(assuming_op, new_assuming_op_results.drop_back());
  rewriter.replaceOp(op, new_assuming_op_results.back());
  return success();
}