TNode<String> CodeStubAssembler::NumberToString(TNode<Number> input) {
  TVARIABLE(String, result);
  TVARIABLE(Smi, smi_input);
  Label runtime(this, Label::kDeferred), if_smi(this), if_heap_number(this),
      done(this, &result);

  // Load the number string cache.
  Node* number_string_cache = LoadRoot(RootIndex::kNumberStringCache);

  // Make the hash mask from the length of the number string cache. It
  // contains two elements (number and string) for each cache entry.
  // TODO(ishell): cleanup mask handling.
  Node* mask =
      BitcastTaggedToWord(LoadFixedArrayBaseLength(number_string_cache));
  TNode<IntPtrT> one = IntPtrConstant(1);
  mask = IntPtrSub(mask, one);

  GotoIfNot(TaggedIsSmi(input), &if_heap_number);
  smi_input = CAST(input);
  Goto(&if_smi);

  BIND(&if_heap_number);
  {
    TNode<HeapNumber> heap_number_input = CAST(input);
    // Try normalizing the HeapNumber.
    TryHeapNumberToSmi(heap_number_input, smi_input, &if_smi);

    // Make a hash from the two 32-bit values of the double.
    TNode<Int32T> low =
        LoadObjectField<Int32T>(heap_number_input, HeapNumber::kValueOffset);
    TNode<Int32T> high = LoadObjectField<Int32T>(
        heap_number_input, HeapNumber::kValueOffset + kIntSize);
    TNode<Word32T> hash = Word32Xor(low, high);
    TNode<WordT> word_hash = WordShl(ChangeInt32ToIntPtr(hash), one);
    TNode<WordT> index =
        WordAnd(word_hash, WordSar(mask, SmiShiftBitsConstant()));

    // Cache entry's key must be a heap number
    Node* number_key =
        UnsafeLoadFixedArrayElement(CAST(number_string_cache), index);
    GotoIf(TaggedIsSmi(number_key), &runtime);
    GotoIfNot(IsHeapNumber(number_key), &runtime);

    // Cache entry's key must match the heap number value we're looking for.
    Node* low_compare = LoadObjectField(number_key, HeapNumber::kValueOffset,
                                        MachineType::Int32());
    Node* high_compare = LoadObjectField(
        number_key, HeapNumber::kValueOffset + kIntSize, MachineType::Int32());
    GotoIfNot(Word32Equal(low, low_compare), &runtime);
    GotoIfNot(Word32Equal(high, high_compare), &runtime);

    // Heap number match, return value from cache entry.
    result = CAST(UnsafeLoadFixedArrayElement(CAST(number_string_cache), index,
                                              kTaggedSize));
    Goto(&done);
  }

  BIND(&if_smi);
  {
    // Load the smi key, make sure it matches the smi we're looking for.
    Node* smi_index = BitcastWordToTagged(
        WordAnd(WordShl(BitcastTaggedToWord(smi_input.value()), one), mask));
    Node* smi_key = UnsafeLoadFixedArrayElement(CAST(number_string_cache),
                                                smi_index, 0, SMI_PARAMETERS);
    GotoIf(WordNotEqual(smi_key, smi_input.value()), &runtime);

    // Smi match, return value from cache entry.
    result = CAST(UnsafeLoadFixedArrayElement(
        CAST(number_string_cache), smi_index, kTaggedSize, SMI_PARAMETERS));
    Goto(&done);
  }

  BIND(&runtime);
  {
    // No cache entry, go to the runtime.
    result =
        CAST(CallRuntime(Runtime::kNumberToString, NoContextConstant(), input));
    Goto(&done);
  }
  BIND(&done);
  return result.value();
}
