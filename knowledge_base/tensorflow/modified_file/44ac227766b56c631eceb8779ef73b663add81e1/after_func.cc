    // Set attr if it is a shape Op
    if (is_shape_calc_op) {
      if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(), true_attr_);
        op->setAttr(kDiscShapeCalcAttr, ArrayAttr::get(tp.getContext(), attrs));
      } else {
        op->setAttr(kDiscShapeCalcAttr, true_attr_);
      }
    }
    return;
  });
}

void MarkShapeCalc::markI64ReturnedCpuScalarOps(
    FuncOp func, llvm::DenseSet<Operation*>& shape_calc_ops) {
  assert(func.getName() == "main");
  auto* return_op = func.front().getTerminator();
  if (!isa<mlir::ReturnOp>(return_op)) return;
  auto result_attrs = func.getAllResultAttrs();
  if (!result_attrs) return;
  auto returned_ops = return_op->getOperands();
