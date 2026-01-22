void Builtins::Generate_CompileLazy(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- eax : argument count (preserved for callee)
  //  -- edx : new target (preserved for callee)
  //  -- edi : target function (preserved for callee)
  // -----------------------------------
  // First lookup code, maybe we don't need to compile!
  Label gotta_call_runtime, gotta_call_runtime_no_stack;
  Label maybe_call_runtime;
  Label try_shared;
  Label loop_top, loop_bottom;

  Register closure = edi;
  Register new_target = edx;
  Register argument_count = eax;

  __ push(argument_count);
  __ push(new_target);
  __ push(closure);

  Register map = argument_count;
  Register index = ebx;
  __ mov(map, FieldOperand(closure, JSFunction::kSharedFunctionInfoOffset));
  __ mov(map, FieldOperand(map, SharedFunctionInfo::kOptimizedCodeMapOffset));
  __ mov(index, FieldOperand(map, FixedArray::kLengthOffset));
  __ cmp(index, Immediate(Smi::FromInt(2)));
  __ j(less, &gotta_call_runtime);

  // Find literals.
  // edx : native context
  // ebx : length / index
  // eax : optimized code map
  // stack[0] : new target
  // stack[4] : closure
  Register native_context = edx;
  __ mov(native_context, NativeContextOperand());

  __ bind(&loop_top);
  Register temp = edi;

  // Does the native context match?
  __ mov(temp, FieldOperand(map, index, times_half_pointer_size,
                            SharedFunctionInfo::kOffsetToPreviousContext));
  __ mov(temp, FieldOperand(temp, WeakCell::kValueOffset));
  __ cmp(temp, native_context);
  __ j(not_equal, &loop_bottom);
  // OSR id set to none?
  __ mov(temp, FieldOperand(map, index, times_half_pointer_size,
                            SharedFunctionInfo::kOffsetToPreviousOsrAstId));
  const int bailout_id = BailoutId::None().ToInt();
  __ cmp(temp, Immediate(Smi::FromInt(bailout_id)));
  __ j(not_equal, &loop_bottom);

  // Literals available?
  Label got_literals, maybe_cleared_weakcell;
  __ mov(temp, FieldOperand(map, index, times_half_pointer_size,
                            SharedFunctionInfo::kOffsetToPreviousLiterals));

  // temp contains either a WeakCell pointing to the literals array or the
  // literals array directly.
  STATIC_ASSERT(WeakCell::kValueOffset == FixedArray::kLengthOffset);
  __ JumpIfSmi(FieldOperand(temp, WeakCell::kValueOffset),
               &maybe_cleared_weakcell);
  // The WeakCell value is a pointer, therefore it's a valid literals array.
  __ mov(temp, FieldOperand(temp, WeakCell::kValueOffset));
  __ jmp(&got_literals);

  // We have a smi. If it's 0, then we are looking at a cleared WeakCell
  // around the literals array, and we should visit the runtime. If it's > 0,
  // then temp already contains the literals array.
  __ bind(&maybe_cleared_weakcell);
  __ cmp(FieldOperand(temp, WeakCell::kValueOffset), Immediate(0));
  __ j(equal, &gotta_call_runtime);

  // Save the literals in the closure.
  __ bind(&got_literals);
  __ mov(ecx, Operand(esp, 0));
  __ mov(FieldOperand(ecx, JSFunction::kLiteralsOffset), temp);
  __ push(index);
  __ RecordWriteField(ecx, JSFunction::kLiteralsOffset, temp, index,
                      kDontSaveFPRegs, EMIT_REMEMBERED_SET, OMIT_SMI_CHECK);
  __ pop(index);

  // Code available?
  Register entry = ecx;
  __ mov(entry, FieldOperand(map, index, times_half_pointer_size,
                             SharedFunctionInfo::kOffsetToPreviousCachedCode));
  __ mov(entry, FieldOperand(entry, WeakCell::kValueOffset));
  __ JumpIfSmi(entry, &maybe_call_runtime);

  // Found literals and code. Get them into the closure and return.
  __ pop(closure);
  // Store code entry in the closure.
  __ lea(entry, FieldOperand(entry, Code::kHeaderSize));

  Label install_optimized_code_and_tailcall;
  __ bind(&install_optimized_code_and_tailcall);
  __ mov(FieldOperand(closure, JSFunction::kCodeEntryOffset), entry);
  __ RecordWriteCodeEntryField(closure, entry, eax);

  // Link the closure into the optimized function list.
  // ecx : code entry
  // edx : native context
  // edi : closure
  __ mov(ebx,
         ContextOperand(native_context, Context::OPTIMIZED_FUNCTIONS_LIST));
  __ mov(FieldOperand(closure, JSFunction::kNextFunctionLinkOffset), ebx);
  __ RecordWriteField(closure, JSFunction::kNextFunctionLinkOffset, ebx, eax,
                      kDontSaveFPRegs, EMIT_REMEMBERED_SET, OMIT_SMI_CHECK);
  const int function_list_offset =
      Context::SlotOffset(Context::OPTIMIZED_FUNCTIONS_LIST);
  __ mov(ContextOperand(native_context, Context::OPTIMIZED_FUNCTIONS_LIST),
         closure);
  // Save closure before the write barrier.
  __ mov(ebx, closure);
  __ RecordWriteContextSlot(native_context, function_list_offset, closure, eax,
                            kDontSaveFPRegs);
  __ mov(closure, ebx);
  __ pop(new_target);
  __ pop(argument_count);
  __ jmp(entry);

  __ bind(&loop_bottom);
  __ sub(index, Immediate(Smi::FromInt(SharedFunctionInfo::kEntryLength)));
  __ cmp(index, Immediate(Smi::FromInt(1)));
  __ j(greater, &loop_top);

  // We found neither literals nor code.
  __ jmp(&gotta_call_runtime);

  __ bind(&maybe_call_runtime);
  __ pop(closure);

  // Last possibility. Check the context free optimized code map entry.
  __ mov(entry, FieldOperand(map, FixedArray::kHeaderSize +
                                      SharedFunctionInfo::kSharedCodeIndex));
  __ mov(entry, FieldOperand(entry, WeakCell::kValueOffset));
  __ JumpIfSmi(entry, &try_shared);

  // Store code entry in the closure.
  __ lea(entry, FieldOperand(entry, Code::kHeaderSize));
  __ jmp(&install_optimized_code_and_tailcall);

  __ bind(&try_shared);
  __ pop(new_target);
  __ pop(argument_count);
  // Is the full code valid?
  __ mov(entry, FieldOperand(closure, JSFunction::kSharedFunctionInfoOffset));
  __ mov(entry, FieldOperand(entry, SharedFunctionInfo::kCodeOffset));
  __ mov(ebx, FieldOperand(entry, Code::kFlagsOffset));
  __ and_(ebx, Code::KindField::kMask);
  __ shr(ebx, Code::KindField::kShift);
  __ cmp(ebx, Immediate(Code::BUILTIN));
  __ j(equal, &gotta_call_runtime_no_stack);
  // Yes, install the full code.
  __ lea(entry, FieldOperand(entry, Code::kHeaderSize));
  __ mov(FieldOperand(closure, JSFunction::kCodeEntryOffset), entry);
  __ RecordWriteCodeEntryField(closure, entry, ebx);
  __ jmp(entry);

  __ bind(&gotta_call_runtime);
  __ pop(closure);
  __ pop(new_target);
  __ pop(argument_count);
  __ bind(&gotta_call_runtime_no_stack);

  GenerateTailCallToReturnedCode(masm, Runtime::kCompileLazy);
}
