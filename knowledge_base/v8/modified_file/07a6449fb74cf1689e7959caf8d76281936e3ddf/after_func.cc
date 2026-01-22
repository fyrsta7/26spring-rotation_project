void CompareICStub::GenerateGeneric(MacroAssembler* masm) {
  Label runtime_call, check_unequal_objects;
  Condition cc = GetCondition();

  Label miss;
  CheckInputType(masm, edx, left(), &miss);
  CheckInputType(masm, eax, right(), &miss);

  // Compare two smis.
  Label non_smi, smi_done;
  __ mov(ecx, edx);
  __ or_(ecx, eax);
  __ JumpIfNotSmi(ecx, &non_smi, Label::kNear);
  __ sub(edx, eax);  // Return on the result of the subtraction.
  __ j(no_overflow, &smi_done, Label::kNear);
  __ not_(edx);  // Correct sign in case of overflow. edx is never 0 here.
  __ bind(&smi_done);
  __ mov(eax, edx);
  __ ret(0);
  __ bind(&non_smi);

  // NOTICE! This code is only reached after a smi-fast-case check, so
  // it is certain that at least one operand isn't a smi.

  // Identical objects can be compared fast, but there are some tricky cases
  // for NaN and undefined.
  Label generic_heap_number_comparison;
  {
    Label not_identical;
    __ cmp(eax, edx);
    __ j(not_equal, &not_identical);

    if (cc != equal) {
      // Check for undefined.  undefined OP undefined is false even though
      // undefined == undefined.
      __ cmp(edx, isolate()->factory()->undefined_value());
      Label check_for_nan;
      __ j(not_equal, &check_for_nan, Label::kNear);
      __ Move(eax, Immediate(Smi::FromInt(NegativeComparisonResult(cc))));
      __ ret(0);
      __ bind(&check_for_nan);
    }

    // Test for NaN. Compare heap numbers in a general way,
    // to handle NaNs correctly.
    __ cmp(FieldOperand(edx, HeapObject::kMapOffset),
           Immediate(isolate()->factory()->heap_number_map()));
    __ j(equal, &generic_heap_number_comparison, Label::kNear);
    if (cc != equal) {
      __ mov(ecx, FieldOperand(eax, HeapObject::kMapOffset));
      __ movzx_b(ecx, FieldOperand(ecx, Map::kInstanceTypeOffset));
      // Call runtime on identical JSObjects.  Otherwise return equal.
      __ cmpb(ecx, Immediate(FIRST_JS_RECEIVER_TYPE));
      __ j(above_equal, &runtime_call, Label::kFar);
      // Call runtime on identical symbols since we need to throw a TypeError.
      __ cmpb(ecx, Immediate(SYMBOL_TYPE));
      __ j(equal, &runtime_call, Label::kFar);
      // Call runtime on identical SIMD values since we must throw a TypeError.
      __ cmpb(ecx, Immediate(SIMD128_VALUE_TYPE));
      __ j(equal, &runtime_call, Label::kFar);
    }
    __ Move(eax, Immediate(Smi::FromInt(EQUAL)));
    __ ret(0);


    __ bind(&not_identical);
  }

  // Strict equality can quickly decide whether objects are equal.
  // Non-strict object equality is slower, so it is handled later in the stub.
  if (cc == equal && strict()) {
    Label slow;  // Fallthrough label.
    Label not_smis;
    // If we're doing a strict equality comparison, we don't have to do
    // type conversion, so we generate code to do fast comparison for objects
    // and oddballs. Non-smi numbers and strings still go through the usual
    // slow-case code.
    // If either is a Smi (we know that not both are), then they can only
    // be equal if the other is a HeapNumber. If so, use the slow case.
    STATIC_ASSERT(kSmiTag == 0);
    DCHECK_EQ(static_cast<Smi*>(0), Smi::kZero);
    __ mov(ecx, Immediate(kSmiTagMask));
    __ and_(ecx, eax);
    __ test(ecx, edx);
    __ j(not_zero, &not_smis, Label::kNear);
    // One operand is a smi.

    // Check whether the non-smi is a heap number.
    STATIC_ASSERT(kSmiTagMask == 1);
    // ecx still holds eax & kSmiTag, which is either zero or one.
    __ sub(ecx, Immediate(0x01));
    __ mov(ebx, edx);
    __ xor_(ebx, eax);
    __ and_(ebx, ecx);  // ebx holds either 0 or eax ^ edx.
    __ xor_(ebx, eax);
    // if eax was smi, ebx is now edx, else eax.

    // Check if the non-smi operand is a heap number.
    __ cmp(FieldOperand(ebx, HeapObject::kMapOffset),
           Immediate(isolate()->factory()->heap_number_map()));
    // If heap number, handle it in the slow case.
    __ j(equal, &slow, Label::kNear);
    // Return non-equal (ebx is not zero)
    __ mov(eax, ebx);
    __ ret(0);

    __ bind(&not_smis);
    // If either operand is a JSObject or an oddball value, then they are not
    // equal since their pointers are different
    // There is no test for undetectability in strict equality.

    // Get the type of the first operand.
    // If the first object is a JS object, we have done pointer comparison.
    Label first_non_object;
    STATIC_ASSERT(LAST_TYPE == LAST_JS_RECEIVER_TYPE);
    __ CmpObjectType(eax, FIRST_JS_RECEIVER_TYPE, ecx);
    __ j(below, &first_non_object, Label::kNear);

    // Return non-zero (eax is not zero)
    Label return_not_equal;
    STATIC_ASSERT(kHeapObjectTag != 0);
    __ bind(&return_not_equal);
    __ ret(0);

    __ bind(&first_non_object);
    // Check for oddballs: true, false, null, undefined.
    __ CmpInstanceType(ecx, ODDBALL_TYPE);
    __ j(equal, &return_not_equal);

    __ CmpObjectType(edx, FIRST_JS_RECEIVER_TYPE, ecx);
    __ j(above_equal, &return_not_equal);

    // Check for oddballs: true, false, null, undefined.
    __ CmpInstanceType(ecx, ODDBALL_TYPE);
    __ j(equal, &return_not_equal);

    // Fall through to the general case.
    __ bind(&slow);
  }

  // Generate the number comparison code.
  Label non_number_comparison;
  Label unordered;
  __ bind(&generic_heap_number_comparison);
  FloatingPointHelper::CheckFloatOperands(
      masm, &non_number_comparison, ebx);
  FloatingPointHelper::LoadFloatOperand(masm, eax);
  FloatingPointHelper::LoadFloatOperand(masm, edx);
  __ FCmp();

  // Don't base result on EFLAGS when a NaN is involved.
  __ j(parity_even, &unordered, Label::kNear);

  Label below_label, above_label;
  // Return a result of -1, 0, or 1, based on EFLAGS.
  __ j(below, &below_label, Label::kNear);
  __ j(above, &above_label, Label::kNear);

  __ Move(eax, Immediate(0));
  __ ret(0);

  __ bind(&below_label);
  __ mov(eax, Immediate(Smi::FromInt(-1)));
  __ ret(0);

  __ bind(&above_label);
  __ mov(eax, Immediate(Smi::FromInt(1)));
  __ ret(0);

  // If one of the numbers was NaN, then the result is always false.
  // The cc is never not-equal.
  __ bind(&unordered);
  DCHECK(cc != not_equal);
  if (cc == less || cc == less_equal) {
    __ mov(eax, Immediate(Smi::FromInt(1)));
  } else {
    __ mov(eax, Immediate(Smi::FromInt(-1)));
  }
  __ ret(0);

  // The number comparison code did not provide a valid result.
  __ bind(&non_number_comparison);

  // Fast negative check for internalized-to-internalized equality.
  Label check_for_strings;
  if (cc == equal) {
    BranchIfNotInternalizedString(masm, &check_for_strings, eax, ecx);
    BranchIfNotInternalizedString(masm, &check_for_strings, edx, ecx);

    // We've already checked for object identity, so if both operands
    // are internalized they aren't equal. Register eax already holds a
    // non-zero value, which indicates not equal, so just return.
    __ ret(0);
  }

  __ bind(&check_for_strings);

  __ JumpIfNotBothSequentialOneByteStrings(edx, eax, ecx, ebx,
                                           &check_unequal_objects);

  // Inline comparison of one-byte strings.
  if (cc == equal) {
    StringHelper::GenerateFlatOneByteStringEquals(masm, edx, eax, ecx, ebx);
  } else {
    StringHelper::GenerateCompareFlatOneByteStrings(masm, edx, eax, ecx, ebx,
                                                    edi);
  }
#ifdef DEBUG
  __ Abort(kUnexpectedFallThroughFromStringComparison);
#endif

  __ bind(&check_unequal_objects);
  if (cc == equal && !strict()) {
    // Non-strict equality.  Objects are unequal if
    // they are both JSObjects and not undetectable,
    // and their pointers are different.
    Label return_equal, return_unequal, undetectable;
    // At most one is a smi, so we can test for smi by adding the two.
    // A smi plus a heap object has the low bit set, a heap object plus
    // a heap object has the low bit clear.
    STATIC_ASSERT(kSmiTag == 0);
    STATIC_ASSERT(kSmiTagMask == 1);
    __ lea(ecx, Operand(eax, edx, times_1, 0));
    __ test(ecx, Immediate(kSmiTagMask));
    __ j(not_zero, &runtime_call);

    __ mov(ecx, FieldOperand(eax, HeapObject::kMapOffset));
    __ mov(ebx, FieldOperand(edx, HeapObject::kMapOffset));

    __ test_b(FieldOperand(ebx, Map::kBitFieldOffset),
              Immediate(1 << Map::kIsUndetectable));
    __ j(not_zero, &undetectable, Label::kNear);
    __ test_b(FieldOperand(ecx, Map::kBitFieldOffset),
              Immediate(1 << Map::kIsUndetectable));
    __ j(not_zero, &return_unequal, Label::kNear);

    __ CmpInstanceType(ebx, FIRST_JS_RECEIVER_TYPE);
    __ j(below, &runtime_call, Label::kNear);
    __ CmpInstanceType(ecx, FIRST_JS_RECEIVER_TYPE);
    __ j(below, &runtime_call, Label::kNear);

    __ bind(&return_unequal);
    // Return non-equal by returning the non-zero object pointer in eax.
    __ ret(0);  // eax, edx were pushed

    __ bind(&undetectable);
    __ test_b(FieldOperand(ecx, Map::kBitFieldOffset),
              Immediate(1 << Map::kIsUndetectable));
    __ j(zero, &return_unequal, Label::kNear);

    // If both sides are JSReceivers, then the result is false according to
    // the HTML specification, which says that only comparisons with null or
    // undefined are affected by special casing for document.all.
    __ CmpInstanceType(ebx, ODDBALL_TYPE);
    __ j(zero, &return_equal, Label::kNear);
    __ CmpInstanceType(ecx, ODDBALL_TYPE);
    __ j(not_zero, &return_unequal, Label::kNear);

    __ bind(&return_equal);
    __ Move(eax, Immediate(EQUAL));
    __ ret(0);  // eax, edx were pushed
  }
  __ bind(&runtime_call);

  if (cc == equal) {
    {
      FrameScope scope(masm, StackFrame::INTERNAL);
      __ Push(esi);
      __ Call(strict() ? isolate()->builtins()->StrictEqual()
                       : isolate()->builtins()->Equal(),
              RelocInfo::CODE_TARGET);
      __ Pop(esi);
    }
    // Turn true into 0 and false into some non-zero value.
    STATIC_ASSERT(EQUAL == 0);
    __ sub(eax, Immediate(isolate()->factory()->true_value()));
    __ Ret();
  } else {
    // Push arguments below the return address.
    __ pop(ecx);
    __ push(edx);
    __ push(eax);
    __ push(Immediate(Smi::FromInt(NegativeComparisonResult(cc))));

    // Restore return address on the stack.
    __ push(ecx);
    // Call the native; it returns -1 (less), 0 (equal), or 1 (greater)
    // tagged as a small integer.
    __ TailCallRuntime(Runtime::kCompare);
  }

  __ bind(&miss);
  GenerateMiss(masm);
}
