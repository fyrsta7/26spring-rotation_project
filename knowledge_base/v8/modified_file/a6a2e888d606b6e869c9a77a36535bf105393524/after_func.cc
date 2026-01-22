void CodeStubAssembler::FastCheck(TNode<BoolT> condition) {
  Label ok(this), not_ok(this, Label::kDeferred);
  Branch(condition, &ok, &not_ok);
  BIND(&not_ok);
  Unreachable();
  BIND(&ok);
}
