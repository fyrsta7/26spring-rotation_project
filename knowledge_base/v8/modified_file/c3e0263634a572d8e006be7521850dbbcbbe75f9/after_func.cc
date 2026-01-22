  TNode<Number> FlattenIntoArray(
      TNode<Context> context, TNode<JSReceiver> target,
      TNode<JSReceiver> source, TNode<Number> source_length,
      TNode<Number> start, TNode<Number> depth,
      base::Optional<TNode<HeapObject>> mapper_function = base::nullopt,
      base::Optional<TNode<Object>> this_arg = base::nullopt) {
    CSA_DCHECK(this, IsNumberPositive(source_length));
    CSA_DCHECK(this, IsNumberPositive(start));

    // 1. Let targetIndex be start.
    TVARIABLE(Number, var_target_index, start);

    // 2. Let sourceIndex be 0.
    TVARIABLE(Number, var_source_index, SmiConstant(0));

    // 3. Repeat...
    Label loop(this, {&var_target_index, &var_source_index}), done_loop(this);
    Goto(&loop);
    BIND(&loop);
    {
      TNode<Number> source_index = var_source_index.value();
      TNode<Number> target_index = var_target_index.value();

      // ...while sourceIndex < sourceLen
      GotoIfNumberGreaterThanOrEqual(source_index, source_length, &done_loop);

      // a. Let P be ! ToString(sourceIndex).
      // b. Let exists be ? HasProperty(source, P).
      CSA_DCHECK(this,
                 SmiGreaterThanOrEqual(CAST(source_index), SmiConstant(0)));
      const TNode<Oddball> exists =
          HasProperty(context, source, source_index, kHasProperty);

      // c. If exists is true, then
      Label next(this);
      GotoIfNot(IsTrue(exists), &next);
      {
        // i. Let element be ? Get(source, P).
        TNode<Object> element_maybe_smi =
            GetProperty(context, source, source_index);

        // ii. If mapperFunction is present, then
        if (mapper_function) {
          CSA_DCHECK(this, Word32Or(IsUndefined(mapper_function.value()),
                                    IsCallable(mapper_function.value())));
          DCHECK(this_arg.has_value());

          // 1. Set element to ? Call(mapperFunction, thisArg , « element,
          //                          sourceIndex, source »).
          element_maybe_smi =
              Call(context, mapper_function.value(), this_arg.value(),
                   element_maybe_smi, source_index, source);
        }

        // iii. Let shouldFlatten be false.
        Label if_flatten_array(this), if_flatten_proxy(this, Label::kDeferred),
            if_noflatten(this);
        // iv. If depth > 0, then
        GotoIfNumberGreaterThanOrEqual(SmiConstant(0), depth, &if_noflatten);
        // 1. Set shouldFlatten to ? IsArray(element).
        GotoIf(TaggedIsSmi(element_maybe_smi), &if_noflatten);
        TNode<HeapObject> element = CAST(element_maybe_smi);
        GotoIf(IsJSArray(element), &if_flatten_array);
        GotoIfNot(IsJSProxy(element), &if_noflatten);
        Branch(IsTrue(CallRuntime(Runtime::kArrayIsArray, context, element)),
               &if_flatten_proxy, &if_noflatten);

        BIND(&if_flatten_array);
        {
          CSA_DCHECK(this, IsJSArray(element));

          // 1. Let elementLen be ? ToLength(? Get(element, "length")).
          const TNode<Object> element_length =
              LoadObjectField(element, JSArray::kLengthOffset);

          // 2. Set targetIndex to ? FlattenIntoArray(target, element,
          //                                          elementLen, targetIndex,
          //                                          depth - 1).
          var_target_index = CAST(
              CallBuiltin(Builtin::kFlattenIntoArray, context, target, element,
                          element_length, target_index, NumberDec(depth)));
          Goto(&next);
        }

        BIND(&if_flatten_proxy);
        {
          CSA_DCHECK(this, IsJSProxy(element));

          // 1. Let elementLen be ? ToLength(? Get(element, "length")).
          const TNode<Number> element_length = ToLength_Inline(
              context, GetProperty(context, element, LengthStringConstant()));

          // 2. Set targetIndex to ? FlattenIntoArray(target, element,
          //                                          elementLen, targetIndex,
          //                                          depth - 1).
          var_target_index = CAST(
              CallBuiltin(Builtin::kFlattenIntoArray, context, target, element,
                          element_length, target_index, NumberDec(depth)));
          Goto(&next);
        }

        BIND(&if_noflatten);
        {
          // 1. If targetIndex >= 2^53-1, throw a TypeError exception.
          Label throw_error(this, Label::kDeferred);
          GotoIfNumberGreaterThanOrEqual(
              target_index, NumberConstant(kMaxSafeInteger), &throw_error);

          // 2. Perform ? CreateDataPropertyOrThrow(target,
          //                                        ! ToString(targetIndex),
          //                                        element).
          CallBuiltin(Builtin::kFastCreateDataProperty, context, target,
                      target_index, element);

          // 3. Increase targetIndex by 1.
          var_target_index = NumberInc(target_index);
          Goto(&next);

          BIND(&throw_error);
          ThrowTypeError(context, MessageTemplate::kFlattenPastSafeLength,
                         source_length, target_index);
        }
      }
      BIND(&next);

      // d. Increase sourceIndex by 1.
      var_source_index = NumberInc(source_index);
      Goto(&loop);
    }

    BIND(&done_loop);
    return var_target_index.value();
  }
