CodeGenerator::CodeGenResult CodeGenerator::AssembleArchInstruction(
    Instruction* instr) {
  X87OperandConverter i(this, instr);
  InstructionCode opcode = instr->opcode();
  ArchOpcode arch_opcode = ArchOpcodeField::decode(opcode);

  switch (arch_opcode) {
    case kArchCallCodeObject: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      EnsureSpaceForLazyDeopt();
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = Handle<Code>::cast(i.InputHeapObject(0));
        __ call(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        __ add(reg, Immediate(Code::kHeaderSize - kHeapObjectTag));
        __ call(reg);
      }
      RecordCallPosition(instr);
      bool double_result =
          instr->HasOutput() && instr->Output()->IsFPRegister();
      if (double_result) {
        __ lea(esp, Operand(esp, -kDoubleSize));
        __ fstp_d(Operand(esp, 0));
      }
      __ fninit();
      if (double_result) {
        __ fld_d(Operand(esp, 0));
        __ lea(esp, Operand(esp, kDoubleSize));
      } else {
        __ fld1();
      }
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchTailCallCodeObjectFromJSFunction:
    case kArchTailCallCodeObject: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      if (arch_opcode == kArchTailCallCodeObjectFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         no_reg, no_reg, no_reg);
      }
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = Handle<Code>::cast(i.InputHeapObject(0));
        __ jmp(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        __ add(reg, Immediate(Code::kHeaderSize - kHeapObjectTag));
        __ jmp(reg);
      }
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchTailCallAddress: {
      CHECK(!HasImmediateInput(instr, 0));
      Register reg = i.InputRegister(0);
      __ jmp(reg);
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchCallJSFunction: {
      EnsureSpaceForLazyDeopt();
      Register func = i.InputRegister(0);
      if (FLAG_debug_code) {
        // Check the function's context matches the context argument.
        __ cmp(esi, FieldOperand(func, JSFunction::kContextOffset));
        __ Assert(equal, kWrongFunctionContext);
      }
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ call(FieldOperand(func, JSFunction::kCodeEntryOffset));
      RecordCallPosition(instr);
      bool double_result =
          instr->HasOutput() && instr->Output()->IsFPRegister();
      if (double_result) {
        __ lea(esp, Operand(esp, -kDoubleSize));
        __ fstp_d(Operand(esp, 0));
      }
      __ fninit();
      if (double_result) {
        __ fld_d(Operand(esp, 0));
        __ lea(esp, Operand(esp, kDoubleSize));
      } else {
        __ fld1();
      }
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchTailCallJSFunctionFromJSFunction:
    case kArchTailCallJSFunction: {
      Register func = i.InputRegister(0);
      if (FLAG_debug_code) {
        // Check the function's context matches the context argument.
        __ cmp(esi, FieldOperand(func, JSFunction::kContextOffset));
        __ Assert(equal, kWrongFunctionContext);
      }
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      if (arch_opcode == kArchTailCallJSFunctionFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         no_reg, no_reg, no_reg);
      }
      __ jmp(FieldOperand(func, JSFunction::kCodeEntryOffset));
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchPrepareCallCFunction: {
      // Frame alignment requires using FP-relative frame addressing.
      frame_access_state()->SetFrameAccessToFP();
      int const num_parameters = MiscField::decode(instr->opcode());
      __ PrepareCallCFunction(num_parameters, i.TempRegister(0));
      break;
    }
    case kArchPrepareTailCall:
      AssemblePrepareTailCall();
      break;
    case kArchCallCFunction: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      int const num_parameters = MiscField::decode(instr->opcode());
      if (HasImmediateInput(instr, 0)) {
        ExternalReference ref = i.InputExternalReference(0);
        __ CallCFunction(ref, num_parameters);
      } else {
        Register func = i.InputRegister(0);
        __ CallCFunction(func, num_parameters);
      }
      bool double_result =
          instr->HasOutput() && instr->Output()->IsFPRegister();
      if (double_result) {
        __ lea(esp, Operand(esp, -kDoubleSize));
        __ fstp_d(Operand(esp, 0));
      }
      __ fninit();
      if (double_result) {
        __ fld_d(Operand(esp, 0));
        __ lea(esp, Operand(esp, kDoubleSize));
      } else {
        __ fld1();
      }
      frame_access_state()->SetFrameAccessToDefault();
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchJmp:
      AssembleArchJump(i.InputRpo(0));
      break;
    case kArchLookupSwitch:
      AssembleArchLookupSwitch(instr);
      break;
    case kArchTableSwitch:
      AssembleArchTableSwitch(instr);
      break;
    case kArchComment: {
      Address comment_string = i.InputExternalReference(0).address();
      __ RecordComment(reinterpret_cast<const char*>(comment_string));
      break;
    }
    case kArchDebugBreak:
      __ int3();
      break;
    case kArchNop:
    case kArchThrowTerminator:
      // don't emit code for nops.
      break;
    case kArchDeoptimize: {
      int deopt_state_id =
          BuildTranslation(instr, -1, 0, OutputFrameStateCombine::Ignore());
      int double_register_param_count = 0;
      int x87_layout = 0;
      for (size_t i = 0; i < instr->InputCount(); i++) {
        if (instr->InputAt(i)->IsFPRegister()) {
          double_register_param_count++;
        }
      }
      // Currently we use only one X87 register. If double_register_param_count
      // is bigger than 1, it means duplicated double register is added to input
      // of this instruction.
      if (double_register_param_count > 0) {
        x87_layout = (0 << 3) | 1;
      }
      // The layout of x87 register stack is loaded on the top of FPU register
      // stack for deoptimization.
      __ push(Immediate(x87_layout));
      __ fild_s(MemOperand(esp, 0));
      __ lea(esp, Operand(esp, kPointerSize));

      Deoptimizer::BailoutType bailout_type =
          Deoptimizer::BailoutType(MiscField::decode(instr->opcode()));
      CodeGenResult result = AssembleDeoptimizerCall(
          deopt_state_id, bailout_type, current_source_position_);
      if (result != kSuccess) return result;
      break;
    }
    case kArchRet:
      AssembleReturn();
      break;
    case kArchFramePointer:
      __ mov(i.OutputRegister(), ebp);
      break;
    case kArchStackPointer:
      __ mov(i.OutputRegister(), esp);
      break;
    case kArchParentFramePointer:
      if (frame_access_state()->has_frame()) {
        __ mov(i.OutputRegister(), Operand(ebp, 0));
      } else {
        __ mov(i.OutputRegister(), ebp);
      }
      break;
    case kArchTruncateDoubleToI: {
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fld_d(i.InputOperand(0));
      }
      __ TruncateX87TOSToI(i.OutputRegister());
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fstp(0);
      }
      break;
    }
    case kArchStoreWithWriteBarrier: {
      RecordWriteMode mode =
          static_cast<RecordWriteMode>(MiscField::decode(instr->opcode()));
      Register object = i.InputRegister(0);
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      Register value = i.InputRegister(index);
      Register scratch0 = i.TempRegister(0);
      Register scratch1 = i.TempRegister(1);
      auto ool = new (zone()) OutOfLineRecordWrite(this, object, operand, value,
                                                   scratch0, scratch1, mode);
      __ mov(operand, value);
      __ CheckPageFlag(object, scratch0,
                       MemoryChunk::kPointersFromHereAreInterestingMask,
                       not_zero, ool->entry());
      __ bind(ool->exit());
      break;
    }
    case kArchStackSlot: {
      FrameOffset offset =
          frame_access_state()->GetFrameOffset(i.InputInt32(0));
      Register base;
      if (offset.from_stack_pointer()) {
        base = esp;
      } else {
        base = ebp;
      }
      __ lea(i.OutputRegister(), Operand(base, offset.offset()));
      break;
    }
    case kIeee754Float64Acos:
      ASSEMBLE_IEEE754_UNOP(acos);
      break;
    case kIeee754Float64Acosh:
      ASSEMBLE_IEEE754_UNOP(acosh);
      break;
    case kIeee754Float64Asin:
      ASSEMBLE_IEEE754_UNOP(asin);
      break;
    case kIeee754Float64Asinh:
      ASSEMBLE_IEEE754_UNOP(asinh);
      break;
    case kIeee754Float64Atan:
      ASSEMBLE_IEEE754_UNOP(atan);
      break;
    case kIeee754Float64Atanh:
      ASSEMBLE_IEEE754_UNOP(atanh);
      break;
    case kIeee754Float64Atan2:
      ASSEMBLE_IEEE754_BINOP(atan2);
      break;
    case kIeee754Float64Cbrt:
      ASSEMBLE_IEEE754_UNOP(cbrt);
      break;
    case kIeee754Float64Cos:
      __ X87SetFPUCW(0x027F);
      ASSEMBLE_IEEE754_UNOP(cos);
      __ X87SetFPUCW(0x037F);
      break;
    case kIeee754Float64Cosh:
      ASSEMBLE_IEEE754_UNOP(cosh);
      break;
    case kIeee754Float64Expm1:
      __ X87SetFPUCW(0x027F);
      ASSEMBLE_IEEE754_UNOP(expm1);
      __ X87SetFPUCW(0x037F);
      break;
    case kIeee754Float64Exp:
      ASSEMBLE_IEEE754_UNOP(exp);
      break;
    case kIeee754Float64Log:
      ASSEMBLE_IEEE754_UNOP(log);
      break;
    case kIeee754Float64Log1p:
      ASSEMBLE_IEEE754_UNOP(log1p);
      break;
    case kIeee754Float64Log2:
      ASSEMBLE_IEEE754_UNOP(log2);
      break;
    case kIeee754Float64Log10:
      ASSEMBLE_IEEE754_UNOP(log10);
      break;
    case kIeee754Float64Pow: {
      // Keep the x87 FPU stack empty before calling stub code
      __ fstp(0);
      // Call the MathStub and put return value in stX_0
      MathPowStub stub(isolate(), MathPowStub::DOUBLE);
      __ CallStub(&stub);
      /* Return value is in st(0) on x87. */
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      break;
    }
    case kIeee754Float64Sin:
      __ X87SetFPUCW(0x027F);
      ASSEMBLE_IEEE754_UNOP(sin);
      __ X87SetFPUCW(0x037F);
      break;
    case kIeee754Float64Sinh:
      ASSEMBLE_IEEE754_UNOP(sinh);
      break;
    case kIeee754Float64Tan:
      __ X87SetFPUCW(0x027F);
      ASSEMBLE_IEEE754_UNOP(tan);
      __ X87SetFPUCW(0x037F);
      break;
    case kIeee754Float64Tanh:
      ASSEMBLE_IEEE754_UNOP(tanh);
      break;
    case kX87Add:
      if (HasImmediateInput(instr, 1)) {
        __ add(i.InputOperand(0), i.InputImmediate(1));
      } else {
        __ add(i.InputRegister(0), i.InputOperand(1));
      }
      break;
    case kX87And:
      if (HasImmediateInput(instr, 1)) {
        __ and_(i.InputOperand(0), i.InputImmediate(1));
      } else {
        __ and_(i.InputRegister(0), i.InputOperand(1));
      }
      break;
    case kX87Cmp:
      ASSEMBLE_COMPARE(cmp);
      break;
    case kX87Cmp16:
      ASSEMBLE_COMPARE(cmpw);
      break;
    case kX87Cmp8:
      ASSEMBLE_COMPARE(cmpb);
      break;
    case kX87Test:
      ASSEMBLE_COMPARE(test);
      break;
    case kX87Test16:
      ASSEMBLE_COMPARE(test_w);
      break;
    case kX87Test8:
      ASSEMBLE_COMPARE(test_b);
      break;
    case kX87Imul:
      if (HasImmediateInput(instr, 1)) {
        __ imul(i.OutputRegister(), i.InputOperand(0), i.InputInt32(1));
      } else {
        __ imul(i.OutputRegister(), i.InputOperand(1));
      }
      break;
    case kX87ImulHigh:
      __ imul(i.InputRegister(1));
      break;
    case kX87UmulHigh:
      __ mul(i.InputRegister(1));
      break;
    case kX87Idiv:
      __ cdq();
      __ idiv(i.InputOperand(1));
      break;
    case kX87Udiv:
      __ Move(edx, Immediate(0));
      __ div(i.InputOperand(1));
      break;
    case kX87Not:
      __ not_(i.OutputOperand());
      break;
    case kX87Neg:
      __ neg(i.OutputOperand());
      break;
    case kX87Or:
      if (HasImmediateInput(instr, 1)) {
        __ or_(i.InputOperand(0), i.InputImmediate(1));
      } else {
        __ or_(i.InputRegister(0), i.InputOperand(1));
      }
      break;
    case kX87Xor:
      if (HasImmediateInput(instr, 1)) {
        __ xor_(i.InputOperand(0), i.InputImmediate(1));
      } else {
        __ xor_(i.InputRegister(0), i.InputOperand(1));
      }
      break;
    case kX87Sub:
      if (HasImmediateInput(instr, 1)) {
        __ sub(i.InputOperand(0), i.InputImmediate(1));
      } else {
        __ sub(i.InputRegister(0), i.InputOperand(1));
      }
      break;
    case kX87Shl:
      if (HasImmediateInput(instr, 1)) {
        __ shl(i.OutputOperand(), i.InputInt5(1));
      } else {
        __ shl_cl(i.OutputOperand());
      }
      break;
    case kX87Shr:
      if (HasImmediateInput(instr, 1)) {
        __ shr(i.OutputOperand(), i.InputInt5(1));
      } else {
        __ shr_cl(i.OutputOperand());
      }
      break;
    case kX87Sar:
      if (HasImmediateInput(instr, 1)) {
        __ sar(i.OutputOperand(), i.InputInt5(1));
      } else {
        __ sar_cl(i.OutputOperand());
      }
      break;
    case kX87AddPair: {
      // i.OutputRegister(0) == i.InputRegister(0) ... left low word.
      // i.InputRegister(1) ... left high word.
      // i.InputRegister(2) ... right low word.
      // i.InputRegister(3) ... right high word.
      bool use_temp = false;
      if (i.OutputRegister(0).code() == i.InputRegister(1).code() ||
          i.OutputRegister(0).code() == i.InputRegister(3).code()) {
        // We cannot write to the output register directly, because it would
        // overwrite an input for adc. We have to use the temp register.
        use_temp = true;
        __ Move(i.TempRegister(0), i.InputRegister(0));
        __ add(i.TempRegister(0), i.InputRegister(2));
      } else {
        __ add(i.OutputRegister(0), i.InputRegister(2));
      }
      __ adc(i.InputRegister(1), Operand(i.InputRegister(3)));
      if (i.OutputRegister(1).code() != i.InputRegister(1).code()) {
        __ Move(i.OutputRegister(1), i.InputRegister(1));
      }
      if (use_temp) {
        __ Move(i.OutputRegister(0), i.TempRegister(0));
      }
      break;
    }
    case kX87SubPair: {
      // i.OutputRegister(0) == i.InputRegister(0) ... left low word.
      // i.InputRegister(1) ... left high word.
      // i.InputRegister(2) ... right low word.
      // i.InputRegister(3) ... right high word.
      bool use_temp = false;
      if (i.OutputRegister(0).code() == i.InputRegister(1).code() ||
          i.OutputRegister(0).code() == i.InputRegister(3).code()) {
        // We cannot write to the output register directly, because it would
        // overwrite an input for adc. We have to use the temp register.
        use_temp = true;
        __ Move(i.TempRegister(0), i.InputRegister(0));
        __ sub(i.TempRegister(0), i.InputRegister(2));
      } else {
        __ sub(i.OutputRegister(0), i.InputRegister(2));
      }
      __ sbb(i.InputRegister(1), Operand(i.InputRegister(3)));
      if (i.OutputRegister(1).code() != i.InputRegister(1).code()) {
        __ Move(i.OutputRegister(1), i.InputRegister(1));
      }
      if (use_temp) {
        __ Move(i.OutputRegister(0), i.TempRegister(0));
      }
      break;
    }
    case kX87MulPair: {
      __ imul(i.OutputRegister(1), i.InputOperand(0));
      __ mov(i.TempRegister(0), i.InputOperand(1));
      __ imul(i.TempRegister(0), i.InputOperand(2));
      __ add(i.OutputRegister(1), i.TempRegister(0));
      __ mov(i.OutputRegister(0), i.InputOperand(0));
      // Multiplies the low words and stores them in eax and edx.
      __ mul(i.InputRegister(2));
      __ add(i.OutputRegister(1), i.TempRegister(0));

      break;
    }
    case kX87ShlPair:
      if (HasImmediateInput(instr, 2)) {
        __ ShlPair(i.InputRegister(1), i.InputRegister(0), i.InputInt6(2));
      } else {
        // Shift has been loaded into CL by the register allocator.
        __ ShlPair_cl(i.InputRegister(1), i.InputRegister(0));
      }
      break;
    case kX87ShrPair:
      if (HasImmediateInput(instr, 2)) {
        __ ShrPair(i.InputRegister(1), i.InputRegister(0), i.InputInt6(2));
      } else {
        // Shift has been loaded into CL by the register allocator.
        __ ShrPair_cl(i.InputRegister(1), i.InputRegister(0));
      }
      break;
    case kX87SarPair:
      if (HasImmediateInput(instr, 2)) {
        __ SarPair(i.InputRegister(1), i.InputRegister(0), i.InputInt6(2));
      } else {
        // Shift has been loaded into CL by the register allocator.
        __ SarPair_cl(i.InputRegister(1), i.InputRegister(0));
      }
      break;
    case kX87Ror:
      if (HasImmediateInput(instr, 1)) {
        __ ror(i.OutputOperand(), i.InputInt5(1));
      } else {
        __ ror_cl(i.OutputOperand());
      }
      break;
    case kX87Lzcnt:
      __ Lzcnt(i.OutputRegister(), i.InputOperand(0));
      break;
    case kX87Popcnt:
      __ Popcnt(i.OutputRegister(), i.InputOperand(0));
      break;
    case kX87LoadFloat64Constant: {
      InstructionOperand* source = instr->InputAt(0);
      InstructionOperand* destination = instr->Output();
      DCHECK(source->IsConstant());
      X87OperandConverter g(this, nullptr);
      Constant src_constant = g.ToConstant(source);

      DCHECK_EQ(Constant::kFloat64, src_constant.type());
      uint64_t src = bit_cast<uint64_t>(src_constant.ToFloat64());
      uint32_t lower = static_cast<uint32_t>(src);
      uint32_t upper = static_cast<uint32_t>(src >> 32);
      if (destination->IsFPRegister()) {
        __ sub(esp, Immediate(kDoubleSize));
        __ mov(MemOperand(esp, 0), Immediate(lower));
        __ mov(MemOperand(esp, kInt32Size), Immediate(upper));
        __ fstp(0);
        __ fld_d(MemOperand(esp, 0));
        __ add(esp, Immediate(kDoubleSize));
      } else {
        UNREACHABLE();
      }
      break;
    }
    case kX87Float32Cmp: {
      __ fld_s(MemOperand(esp, kFloatSize));
      __ fld_s(MemOperand(esp, 0));
      __ FCmp();
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      break;
    }
    case kX87Float32Add: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_s(MemOperand(esp, 0));
      __ fld_s(MemOperand(esp, kFloatSize));
      __ faddp();
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float32Sub: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_s(MemOperand(esp, kFloatSize));
      __ fld_s(MemOperand(esp, 0));
      __ fsubp();
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float32Mul: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_s(MemOperand(esp, kFloatSize));
      __ fld_s(MemOperand(esp, 0));
      __ fmulp();
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float32Div: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_s(MemOperand(esp, kFloatSize));
      __ fld_s(MemOperand(esp, 0));
      __ fdivp();
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }

    case kX87Float32Sqrt: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_s(MemOperand(esp, 0));
      __ fsqrt();
      __ lea(esp, Operand(esp, kFloatSize));
      break;
    }
    case kX87Float32Abs: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_s(MemOperand(esp, 0));
      __ fabs();
      __ lea(esp, Operand(esp, kFloatSize));
      break;
    }
    case kX87Float32Neg: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_s(MemOperand(esp, 0));
      __ fchs();
      __ lea(esp, Operand(esp, kFloatSize));
      break;
    }
    case kX87Float32Round: {
      RoundingMode mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      // Set the correct round mode in x87 control register
      __ X87SetRC((mode << 10));

      if (!instr->InputAt(0)->IsFPRegister()) {
        InstructionOperand* input = instr->InputAt(0);
        USE(input);
        DCHECK(input->IsFPStackSlot());
        if (FLAG_debug_code && FLAG_enable_slow_asserts) {
          __ VerifyX87StackDepth(1);
        }
        __ fstp(0);
        __ fld_s(i.InputOperand(0));
      }
      __ frndint();
      __ X87SetRC(0x0000);
      break;
    }
    case kX87Float64Add: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_d(MemOperand(esp, 0));
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ faddp();
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float64Sub: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ fsub_d(MemOperand(esp, 0));
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float64Mul: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ fmul_d(MemOperand(esp, 0));
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float64Div: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ fdiv_d(MemOperand(esp, 0));
      // Clear stack.
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      // Restore the default value of control word.
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float64Mod: {
      FrameScope frame_scope(&masm_, StackFrame::MANUAL);
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ mov(eax, esp);
      __ PrepareCallCFunction(4, eax);
      __ fstp(0);
      __ fld_d(MemOperand(eax, 0));
      __ fstp_d(Operand(esp, 1 * kDoubleSize));
      __ fld_d(MemOperand(eax, kDoubleSize));
      __ fstp_d(Operand(esp, 0));
      __ CallCFunction(ExternalReference::mod_two_doubles_operation(isolate()),
                       4);
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      break;
    }
    case kX87Float32Max: {
      Label compare_swap, done_compare;
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_s(MemOperand(esp, kFloatSize));
      __ fld_s(MemOperand(esp, 0));
      __ fld(1);
      __ fld(1);
      __ FCmp();

      auto ool =
          new (zone()) OutOfLineLoadFloat32NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(below, &done_compare, Label::kNear);
      __ j(above, &compare_swap, Label::kNear);
      __ push(eax);
      __ lea(esp, Operand(esp, -kFloatSize));
      __ fld(1);
      __ fstp_s(Operand(esp, 0));
      __ mov(eax, MemOperand(esp, 0));
      __ and_(eax, Immediate(0x80000000));
      __ lea(esp, Operand(esp, kFloatSize));
      __ pop(eax);
      __ j(zero, &done_compare, Label::kNear);

      __ bind(&compare_swap);
      __ bind(ool->exit());
      __ fxch(1);

      __ bind(&done_compare);
      __ fstp(0);
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      break;
    }
    case kX87Float64Max: {
      Label compare_swap, done_compare;
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ fld_d(MemOperand(esp, 0));
      __ fld(1);
      __ fld(1);
      __ FCmp();

      auto ool =
          new (zone()) OutOfLineLoadFloat64NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(below, &done_compare, Label::kNear);
      __ j(above, &compare_swap, Label::kNear);
      __ push(eax);
      __ lea(esp, Operand(esp, -kDoubleSize));
      __ fld(1);
      __ fstp_d(Operand(esp, 0));
      __ mov(eax, MemOperand(esp, 4));
      __ and_(eax, Immediate(0x80000000));
      __ lea(esp, Operand(esp, kDoubleSize));
      __ pop(eax);
      __ j(zero, &done_compare, Label::kNear);

      __ bind(&compare_swap);
      __ bind(ool->exit());
      __ fxch(1);

      __ bind(&done_compare);
      __ fstp(0);
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      break;
    }
    case kX87Float32Min: {
      Label compare_swap, done_compare;
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_s(MemOperand(esp, kFloatSize));
      __ fld_s(MemOperand(esp, 0));
      __ fld(1);
      __ fld(1);
      __ FCmp();

      auto ool =
          new (zone()) OutOfLineLoadFloat32NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(above, &done_compare, Label::kNear);
      __ j(below, &compare_swap, Label::kNear);
      __ push(eax);
      __ lea(esp, Operand(esp, -kFloatSize));
      __ fld(0);
      __ fstp_s(Operand(esp, 0));
      __ mov(eax, MemOperand(esp, 0));
      __ and_(eax, Immediate(0x80000000));
      __ lea(esp, Operand(esp, kFloatSize));
      __ pop(eax);
      __ j(zero, &done_compare, Label::kNear);

      __ bind(&compare_swap);
      __ bind(ool->exit());
      __ fxch(1);

      __ bind(&done_compare);
      __ fstp(0);
      __ lea(esp, Operand(esp, 2 * kFloatSize));
      break;
    }
    case kX87Float64Min: {
      Label compare_swap, done_compare;
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ fld_d(MemOperand(esp, 0));
      __ fld(1);
      __ fld(1);
      __ FCmp();

      auto ool =
          new (zone()) OutOfLineLoadFloat64NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(above, &done_compare, Label::kNear);
      __ j(below, &compare_swap, Label::kNear);
      __ push(eax);
      __ lea(esp, Operand(esp, -kDoubleSize));
      __ fld(0);
      __ fstp_d(Operand(esp, 0));
      __ mov(eax, MemOperand(esp, 4));
      __ and_(eax, Immediate(0x80000000));
      __ lea(esp, Operand(esp, kDoubleSize));
      __ pop(eax);
      __ j(zero, &done_compare, Label::kNear);

      __ bind(&compare_swap);
      __ bind(ool->exit());
      __ fxch(1);

      __ bind(&done_compare);
      __ fstp(0);
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      break;
    }
    case kX87Float64Abs: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_d(MemOperand(esp, 0));
      __ fabs();
      __ lea(esp, Operand(esp, kDoubleSize));
      break;
    }
    case kX87Float64Neg: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ fld_d(MemOperand(esp, 0));
      __ fchs();
      __ lea(esp, Operand(esp, kDoubleSize));
      break;
    }
    case kX87Int32ToFloat32: {
      InstructionOperand* input = instr->InputAt(0);
      DCHECK(input->IsRegister() || input->IsStackSlot());
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      if (input->IsRegister()) {
        Register input_reg = i.InputRegister(0);
        __ push(input_reg);
        __ fild_s(Operand(esp, 0));
        __ pop(input_reg);
      } else {
        __ fild_s(i.InputOperand(0));
      }
      break;
    }
    case kX87Uint32ToFloat32: {
      InstructionOperand* input = instr->InputAt(0);
      DCHECK(input->IsRegister() || input->IsStackSlot());
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      Label msb_set_src;
      Label jmp_return;
      // Put input integer into eax(tmporarilly)
      __ push(eax);
      if (input->IsRegister())
        __ mov(eax, i.InputRegister(0));
      else
        __ mov(eax, i.InputOperand(0));

      __ test(eax, eax);
      __ j(sign, &msb_set_src, Label::kNear);
      __ push(eax);
      __ fild_s(Operand(esp, 0));
      __ pop(eax);

      __ jmp(&jmp_return, Label::kNear);
      __ bind(&msb_set_src);
      // Need another temp reg
      __ push(ebx);
      __ mov(ebx, eax);
      __ shr(eax, 1);
      // Recover the least significant bit to avoid rounding errors.
      __ and_(ebx, Immediate(1));
      __ or_(eax, ebx);
      __ push(eax);
      __ fild_s(Operand(esp, 0));
      __ pop(eax);
      __ fld(0);
      __ faddp();
      // Restore the ebx
      __ pop(ebx);
      __ bind(&jmp_return);
      // Restore the eax
      __ pop(eax);
      break;
    }
    case kX87Int32ToFloat64: {
      InstructionOperand* input = instr->InputAt(0);
      DCHECK(input->IsRegister() || input->IsStackSlot());
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      if (input->IsRegister()) {
        Register input_reg = i.InputRegister(0);
        __ push(input_reg);
        __ fild_s(Operand(esp, 0));
        __ pop(input_reg);
      } else {
        __ fild_s(i.InputOperand(0));
      }
      break;
    }
    case kX87Float32ToFloat64: {
      InstructionOperand* input = instr->InputAt(0);
      if (input->IsFPRegister()) {
        __ sub(esp, Immediate(kDoubleSize));
        __ fstp_s(MemOperand(esp, 0));
        __ fld_s(MemOperand(esp, 0));
        __ add(esp, Immediate(kDoubleSize));
      } else {
        DCHECK(input->IsFPStackSlot());
        if (FLAG_debug_code && FLAG_enable_slow_asserts) {
          __ VerifyX87StackDepth(1);
        }
        __ fstp(0);
        __ fld_s(i.InputOperand(0));
      }
      break;
    }
    case kX87Uint32ToFloat64: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      __ LoadUint32NoSSE2(i.InputRegister(0));
      break;
    }
    case kX87Float32ToInt32: {
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fld_s(i.InputOperand(0));
      }
      __ TruncateX87TOSToI(i.OutputRegister(0));
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fstp(0);
      }
      break;
    }
    case kX87Float32ToUint32: {
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fld_s(i.InputOperand(0));
      }
      Label success;
      __ TruncateX87TOSToI(i.OutputRegister(0));
      __ test(i.OutputRegister(0), i.OutputRegister(0));
      __ j(positive, &success);
      // Need to reserve the input float32 data.
      __ fld(0);
      __ push(Immediate(INT32_MIN));
      __ fild_s(Operand(esp, 0));
      __ lea(esp, Operand(esp, kPointerSize));
      __ faddp();
      __ TruncateX87TOSToI(i.OutputRegister(0));
      __ or_(i.OutputRegister(0), Immediate(0x80000000));
      // Only keep input float32 data in x87 stack when return.
      __ fstp(0);
      __ bind(&success);
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fstp(0);
      }
      break;
    }
    case kX87Float64ToInt32: {
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fld_d(i.InputOperand(0));
      }
      __ TruncateX87TOSToI(i.OutputRegister(0));
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fstp(0);
      }
      break;
    }
    case kX87Float64ToFloat32: {
      InstructionOperand* input = instr->InputAt(0);
      if (input->IsFPRegister()) {
        __ sub(esp, Immediate(kDoubleSize));
        __ fstp_s(MemOperand(esp, 0));
        __ fld_s(MemOperand(esp, 0));
        __ add(esp, Immediate(kDoubleSize));
      } else {
        DCHECK(input->IsFPStackSlot());
        if (FLAG_debug_code && FLAG_enable_slow_asserts) {
          __ VerifyX87StackDepth(1);
        }
        __ fstp(0);
        __ fld_d(i.InputOperand(0));
        __ sub(esp, Immediate(kDoubleSize));
        __ fstp_s(MemOperand(esp, 0));
        __ fld_s(MemOperand(esp, 0));
        __ add(esp, Immediate(kDoubleSize));
      }
      break;
    }
    case kX87Float64ToUint32: {
      __ push_imm32(-2147483648);
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fld_d(i.InputOperand(0));
      }
      __ fild_s(Operand(esp, 0));
      __ fld(1);
      __ faddp();
      __ TruncateX87TOSToI(i.OutputRegister(0));
      __ add(esp, Immediate(kInt32Size));
      __ add(i.OutputRegister(), Immediate(0x80000000));
      __ fstp(0);
      if (!instr->InputAt(0)->IsFPRegister()) {
        __ fstp(0);
      }
      break;
    }
    case kX87Float64ExtractHighWord32: {
      if (instr->InputAt(0)->IsFPRegister()) {
        __ sub(esp, Immediate(kDoubleSize));
        __ fst_d(MemOperand(esp, 0));
        __ mov(i.OutputRegister(), MemOperand(esp, kDoubleSize / 2));
        __ add(esp, Immediate(kDoubleSize));
      } else {
        InstructionOperand* input = instr->InputAt(0);
        USE(input);
        DCHECK(input->IsFPStackSlot());
        __ mov(i.OutputRegister(), i.InputOperand(0, kDoubleSize / 2));
      }
      break;
    }
    case kX87Float64ExtractLowWord32: {
      if (instr->InputAt(0)->IsFPRegister()) {
        __ sub(esp, Immediate(kDoubleSize));
        __ fst_d(MemOperand(esp, 0));
        __ mov(i.OutputRegister(), MemOperand(esp, 0));
        __ add(esp, Immediate(kDoubleSize));
      } else {
        InstructionOperand* input = instr->InputAt(0);
        USE(input);
        DCHECK(input->IsFPStackSlot());
        __ mov(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    }
    case kX87Float64InsertHighWord32: {
      __ sub(esp, Immediate(kDoubleSize));
      __ fstp_d(MemOperand(esp, 0));
      __ mov(MemOperand(esp, kDoubleSize / 2), i.InputRegister(1));
      __ fld_d(MemOperand(esp, 0));
      __ add(esp, Immediate(kDoubleSize));
      break;
    }
    case kX87Float64InsertLowWord32: {
      __ sub(esp, Immediate(kDoubleSize));
      __ fstp_d(MemOperand(esp, 0));
      __ mov(MemOperand(esp, 0), i.InputRegister(1));
      __ fld_d(MemOperand(esp, 0));
      __ add(esp, Immediate(kDoubleSize));
      break;
    }
    case kX87Float64Sqrt: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ X87SetFPUCW(0x027F);
      __ fstp(0);
      __ fld_d(MemOperand(esp, 0));
      __ fsqrt();
      __ lea(esp, Operand(esp, kDoubleSize));
      __ X87SetFPUCW(0x037F);
      break;
    }
    case kX87Float64Round: {
      RoundingMode mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      // Set the correct round mode in x87 control register
      __ X87SetRC((mode << 10));

      if (!instr->InputAt(0)->IsFPRegister()) {
        InstructionOperand* input = instr->InputAt(0);
        USE(input);
        DCHECK(input->IsFPStackSlot());
        if (FLAG_debug_code && FLAG_enable_slow_asserts) {
          __ VerifyX87StackDepth(1);
        }
        __ fstp(0);
        __ fld_d(i.InputOperand(0));
      }
      __ frndint();
      __ X87SetRC(0x0000);
      break;
    }
    case kX87Float64Cmp: {
      __ fld_d(MemOperand(esp, kDoubleSize));
      __ fld_d(MemOperand(esp, 0));
      __ FCmp();
      __ lea(esp, Operand(esp, 2 * kDoubleSize));
      break;
    }
    case kX87Float64SilenceNaN: {
      Label end, return_qnan;
      __ fstp(0);
      __ push(ebx);
      // Load Half word of HoleNan(SNaN) into ebx
      __ mov(ebx, MemOperand(esp, 2 * kInt32Size));
      __ cmp(ebx, Immediate(kHoleNanUpper32));
      // Check input is HoleNaN(SNaN)?
      __ j(equal, &return_qnan, Label::kNear);
      // If input isn't HoleNaN(SNaN), just load it and return
      __ fld_d(MemOperand(esp, 1 * kInt32Size));
      __ jmp(&end);
      __ bind(&return_qnan);
      // If input is HoleNaN(SNaN), Return QNaN
      __ push(Immediate(0xffffffff));
      __ push(Immediate(0xfff7ffff));
      __ fld_d(MemOperand(esp, 0));
      __ lea(esp, Operand(esp, kDoubleSize));
      __ bind(&end);
      __ pop(ebx);
      // Clear stack.
      __ lea(esp, Operand(esp, 1 * kDoubleSize));
      break;
    }
    case kX87Movsxbl:
      __ movsx_b(i.OutputRegister(), i.MemoryOperand());
      break;
    case kX87Movzxbl:
      __ movzx_b(i.OutputRegister(), i.MemoryOperand());
      break;
    case kX87Movb: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ mov_b(operand, i.InputInt8(index));
      } else {
        __ mov_b(operand, i.InputRegister(index));
      }
      break;
    }
    case kX87Movsxwl:
      __ movsx_w(i.OutputRegister(), i.MemoryOperand());
      break;
    case kX87Movzxwl:
      __ movzx_w(i.OutputRegister(), i.MemoryOperand());
      break;
    case kX87Movw: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ mov_w(operand, i.InputInt16(index));
      } else {
        __ mov_w(operand, i.InputRegister(index));
      }
      break;
    }
    case kX87Movl:
      if (instr->HasOutput()) {
        __ mov(i.OutputRegister(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        if (HasImmediateInput(instr, index)) {
          __ mov(operand, i.InputImmediate(index));
        } else {
          __ mov(operand, i.InputRegister(index));
        }
      }
      break;
    case kX87Movsd: {
      if (instr->HasOutput()) {
        X87Register output = i.OutputDoubleRegister();
        USE(output);
        DCHECK(output.code() == 0);
        if (FLAG_debug_code && FLAG_enable_slow_asserts) {
          __ VerifyX87StackDepth(1);
        }
        __ fstp(0);
        __ fld_d(i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ fst_d(operand);
      }
      break;
    }
    case kX87Movss: {
      if (instr->HasOutput()) {
        X87Register output = i.OutputDoubleRegister();
        USE(output);
        DCHECK(output.code() == 0);
        if (FLAG_debug_code && FLAG_enable_slow_asserts) {
          __ VerifyX87StackDepth(1);
        }
        __ fstp(0);
        __ fld_s(i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ fst_s(operand);
      }
      break;
    }
    case kX87BitcastFI: {
      __ mov(i.OutputRegister(), MemOperand(esp, 0));
      __ lea(esp, Operand(esp, kFloatSize));
      break;
    }
    case kX87BitcastIF: {
      if (FLAG_debug_code && FLAG_enable_slow_asserts) {
        __ VerifyX87StackDepth(1);
      }
      __ fstp(0);
      if (instr->InputAt(0)->IsRegister()) {
        __ lea(esp, Operand(esp, -kFloatSize));
        __ mov(MemOperand(esp, 0), i.InputRegister(0));
        __ fld_s(MemOperand(esp, 0));
        __ lea(esp, Operand(esp, kFloatSize));
      } else {
        __ fld_s(i.InputOperand(0));
      }
      break;
    }
    case kX87Lea: {
      AddressingMode mode = AddressingModeField::decode(instr->opcode());
      // Shorten "leal" to "addl", "subl" or "shll" if the register allocation
      // and addressing mode just happens to work out. The "addl"/"subl" forms
      // in these cases are faster based on measurements.
      if (mode == kMode_MI) {
        __ Move(i.OutputRegister(), Immediate(i.InputInt32(0)));
      } else if (i.InputRegister(0).is(i.OutputRegister())) {
        if (mode == kMode_MRI) {
          int32_t constant_summand = i.InputInt32(1);
          if (constant_summand > 0) {
            __ add(i.OutputRegister(), Immediate(constant_summand));
          } else if (constant_summand < 0) {
            __ sub(i.OutputRegister(), Immediate(-constant_summand));
          }
        } else if (mode == kMode_MR1) {
          if (i.InputRegister(1).is(i.OutputRegister())) {
            __ shl(i.OutputRegister(), 1);
          } else {
            __ lea(i.OutputRegister(), i.MemoryOperand());
          }
        } else if (mode == kMode_M2) {
          __ shl(i.OutputRegister(), 1);
        } else if (mode == kMode_M4) {
          __ shl(i.OutputRegister(), 2);
        } else if (mode == kMode_M8) {
          __ shl(i.OutputRegister(), 3);
        } else {
          __ lea(i.OutputRegister(), i.MemoryOperand());
        }
      } else {
        __ lea(i.OutputRegister(), i.MemoryOperand());
      }
      break;
    }
    case kX87Push:
      if (instr->InputAt(0)->IsFPRegister()) {
        auto allocated = AllocatedOperand::cast(*instr->InputAt(0));
        if (allocated.representation() == MachineRepresentation::kFloat32) {
          __ sub(esp, Immediate(kFloatSize));
          __ fst_s(Operand(esp, 0));
          frame_access_state()->IncreaseSPDelta(kFloatSize / kPointerSize);
        } else {
          DCHECK(allocated.representation() == MachineRepresentation::kFloat64);
          __ sub(esp, Immediate(kDoubleSize));
          __ fst_d(Operand(esp, 0));
        frame_access_state()->IncreaseSPDelta(kDoubleSize / kPointerSize);
        }
      } else if (instr->InputAt(0)->IsFPStackSlot()) {
        auto allocated = AllocatedOperand::cast(*instr->InputAt(0));
        if (allocated.representation() == MachineRepresentation::kFloat32) {
          __ sub(esp, Immediate(kFloatSize));
          __ fld_s(i.InputOperand(0));
          __ fstp_s(MemOperand(esp, 0));
          frame_access_state()->IncreaseSPDelta(kFloatSize / kPointerSize);
        } else {
          DCHECK(allocated.representation() == MachineRepresentation::kFloat64);
          __ sub(esp, Immediate(kDoubleSize));
          __ fld_d(i.InputOperand(0));
          __ fstp_d(MemOperand(esp, 0));
        frame_access_state()->IncreaseSPDelta(kDoubleSize / kPointerSize);
        }
      } else if (HasImmediateInput(instr, 0)) {
        __ push(i.InputImmediate(0));
        frame_access_state()->IncreaseSPDelta(1);
      } else {
        __ push(i.InputOperand(0));
        frame_access_state()->IncreaseSPDelta(1);
      }
      break;
    case kX87Poke: {
      int const slot = MiscField::decode(instr->opcode());
      if (HasImmediateInput(instr, 0)) {
        __ mov(Operand(esp, slot * kPointerSize), i.InputImmediate(0));
      } else {
        __ mov(Operand(esp, slot * kPointerSize), i.InputRegister(0));
      }
      break;
    }
    case kX87Xchgb: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      __ xchg_b(i.InputRegister(index), operand);
      break;
    }
    case kX87Xchgw: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      __ xchg_w(i.InputRegister(index), operand);
      break;
    }
    case kX87Xchgl: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      __ xchg(i.InputRegister(index), operand);
      break;
    }
    case kX87PushFloat32:
      __ lea(esp, Operand(esp, -kFloatSize));
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ fld_s(i.InputOperand(0));
        __ fstp_s(MemOperand(esp, 0));
      } else if (instr->InputAt(0)->IsFPRegister()) {
        __ fst_s(MemOperand(esp, 0));
      } else {
        UNREACHABLE();
      }
      break;
    case kX87PushFloat64:
      __ lea(esp, Operand(esp, -kDoubleSize));
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ fld_d(i.InputOperand(0));
        __ fstp_d(MemOperand(esp, 0));
      } else if (instr->InputAt(0)->IsFPRegister()) {
        __ fst_d(MemOperand(esp, 0));
      } else {
        UNREACHABLE();
      }
      break;
    case kCheckedLoadInt8:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movsx_b);
      break;
    case kCheckedLoadUint8:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movzx_b);
      break;
    case kCheckedLoadInt16:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movsx_w);
      break;
    case kCheckedLoadUint16:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movzx_w);
      break;
    case kCheckedLoadWord32:
      ASSEMBLE_CHECKED_LOAD_INTEGER(mov);
      break;
    case kCheckedLoadFloat32:
      ASSEMBLE_CHECKED_LOAD_FLOAT(fld_s, OutOfLineLoadFloat32NaN);
      break;
    case kCheckedLoadFloat64:
      ASSEMBLE_CHECKED_LOAD_FLOAT(fld_d, OutOfLineLoadFloat64NaN);
      break;
    case kCheckedStoreWord8:
      ASSEMBLE_CHECKED_STORE_INTEGER(mov_b);
      break;
    case kCheckedStoreWord16:
      ASSEMBLE_CHECKED_STORE_INTEGER(mov_w);
      break;
    case kCheckedStoreWord32:
      ASSEMBLE_CHECKED_STORE_INTEGER(mov);
      break;
    case kCheckedStoreFloat32:
      ASSEMBLE_CHECKED_STORE_FLOAT(fst_s);
      break;
    case kCheckedStoreFloat64:
      ASSEMBLE_CHECKED_STORE_FLOAT(fst_d);
      break;
    case kX87StackCheck: {
      ExternalReference const stack_limit =
          ExternalReference::address_of_stack_limit(isolate());
      __ cmp(esp, Operand::StaticVariable(stack_limit));
      break;
    }
    case kCheckedLoadWord64:
    case kCheckedStoreWord64:
      UNREACHABLE();  // currently unsupported checked int64 load/store.
      break;
    case kAtomicLoadInt8:
    case kAtomicLoadUint8:
    case kAtomicLoadInt16:
    case kAtomicLoadUint16:
    case kAtomicLoadWord32:
    case kAtomicStoreWord8:
    case kAtomicStoreWord16:
    case kAtomicStoreWord32:
      UNREACHABLE();  // Won't be generated by instruction selector.
      break;
  }
  return kSuccess;
}  // NOLINT(readability/fn_size)
