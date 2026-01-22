  void GenerateIteratingArrayBuiltinLoopContinuation(
      const CallResultProcessor& processor, const PostLoopAction& action,
      ForEachDirection direction = ForEachDirection::kForward) {
    Label loop(this, {&k_, &a_, &to_});
    Label after_loop(this);
    Goto(&loop);
    BIND(&loop);
    {
      if (direction == ForEachDirection::kForward) {
        // 8. Repeat, while k < len
        GotoUnlessNumberLessThan(k(), len_, &after_loop);
      } else {
        // OR
        // 10. Repeat, while k >= 0
        GotoUnlessNumberLessThan(SmiConstant(-1), k(), &after_loop);
      }

      Label done_element(this, &to_);
      // a. Let Pk be ToString(k).
      Node* p_k = ToString(context(), k());

      // b. Let kPresent be HasProperty(O, Pk).
      // c. ReturnIfAbrupt(kPresent).
      Node* k_present = HasProperty(o(), p_k, context(), kHasProperty);

      // d. If kPresent is true, then
      GotoIf(WordNotEqual(k_present, TrueConstant()), &done_element);

      // i. Let kValue be Get(O, Pk).
      // ii. ReturnIfAbrupt(kValue).
      Node* k_value = GetProperty(context(), o(), k());

      // iii. Let funcResult be Call(callbackfn, T, «kValue, k, O»).
      // iv. ReturnIfAbrupt(funcResult).
      a_.Bind(processor(this, k_value, k()));
      Goto(&done_element);

      BIND(&done_element);

      if (direction == ForEachDirection::kForward) {
        // e. Increase k by 1.
        k_.Bind(NumberInc(k()));
      } else {
        // e. Decrease k by 1.
        k_.Bind(NumberDec(k()));
      }
      Goto(&loop);
    }
    BIND(&after_loop);

    action(this);
    Return(a_.value());
  }
