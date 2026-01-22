static void Generate_PushAppliedArguments(MacroAssembler* masm,
                                          const int argumentsOffset,
                                          const int indexOffset,
                                          const int limitOffset) {
  // Copy all arguments from the array to the stack.
  Label entry, loop;
  Register receiver = LoadDescriptor::ReceiverRegister();
  Register key = LoadDescriptor::NameRegister();
  Register slot = LoadDescriptor::SlotRegister();
  Register vector = LoadWithVectorDescriptor::VectorRegister();
  __ mov(key, Operand(ebp, indexOffset));
  __ jmp(&entry);
  __ bind(&loop);
  __ mov(receiver, Operand(ebp, argumentsOffset));  // load arguments

  // Use inline caching to speed up access to arguments.
  FeedbackVectorSpec spec(0, Code::KEYED_LOAD_IC);
  Handle<TypeFeedbackVector> feedback_vector =
      masm->isolate()->factory()->NewTypeFeedbackVector(&spec);
  int index = feedback_vector->GetIndex(FeedbackVectorICSlot(0));
  __ mov(slot, Immediate(Smi::FromInt(index)));
  __ mov(vector, Immediate(feedback_vector));
  Handle<Code> ic = KeyedLoadICStub(masm->isolate()).GetCode();
  __ call(ic, RelocInfo::CODE_TARGET);
  // It is important that we do not have a test instruction after the
  // call.  A test instruction after the call is used to indicate that
  // we have generated an inline version of the keyed load.  In this
  // case, we know that we are not generating a test instruction next.

  // Push the nth argument.
  __ push(eax);

  // Update the index on the stack and in register key.
  __ mov(key, Operand(ebp, indexOffset));
  __ add(key, Immediate(1 << kSmiTagSize));
  __ mov(Operand(ebp, indexOffset), key);

  __ bind(&entry);
  __ cmp(key, Operand(ebp, limitOffset));
  __ j(not_equal, &loop);

  // On exit, the pushed arguments count is in eax, untagged
  __ Move(eax, key);
  __ SmiUntag(eax);
}
