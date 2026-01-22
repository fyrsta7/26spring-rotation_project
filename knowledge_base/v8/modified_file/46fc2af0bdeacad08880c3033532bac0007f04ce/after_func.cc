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
      // We never have to perform a ToString conversion for Smi keys because
      // they are guaranteed to be stored as elements. We easily hit this case
      // when using any iteration builtin on a dictionary elements Array.
      VARIABLE(p_k, MachineRepresentation::kTagged, k());
      {
        Label continue_with_key(this);
        GotoIf(TaggedIsSmi(p_k.value()), &continue_with_key);
        p_k.Bind(ToString(context(), p_k.value()));
        Goto(&continue_with_key);
        BIND(&continue_with_key);
      }

      // b. Let kPresent be HasProperty(O, Pk).
      // c. ReturnIfAbrupt(kPresent).
      Node* k_present = HasProperty(o(), p_k.value(), context(), kHasProperty);

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
