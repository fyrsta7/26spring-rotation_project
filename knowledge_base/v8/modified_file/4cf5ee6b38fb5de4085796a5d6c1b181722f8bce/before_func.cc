CodeGenerator::CodeGenResult CodeGenerator::AssembleArchInstruction(
    Instruction* instr) {
  X64OperandConverter i(this, instr);
  InstructionCode opcode = instr->opcode();
  ArchOpcode arch_opcode = ArchOpcodeField::decode(opcode);
  switch (arch_opcode) {
    case kArchCallCodeObject: {
      EnsureSpaceForLazyDeopt();
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = Handle<Code>::cast(i.InputHeapObject(0));
        __ Call(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        __ addp(reg, Immediate(Code::kHeaderSize - kHeapObjectTag));
        __ call(reg);
      }
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchTailCallCodeObjectFromJSFunction:
    case kArchTailCallCodeObject: {
      if (arch_opcode == kArchTailCallCodeObjectFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         i.TempRegister(0), i.TempRegister(1),
                                         i.TempRegister(2));
      }
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = Handle<Code>::cast(i.InputHeapObject(0));
        __ jmp(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        __ addp(reg, Immediate(Code::kHeaderSize - kHeapObjectTag));
        __ jmp(reg);
      }
      unwinding_info_writer_.MarkBlockWillExit();
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchTailCallAddress: {
      CHECK(!HasImmediateInput(instr, 0));
      Register reg = i.InputRegister(0);
      __ jmp(reg);
      unwinding_info_writer_.MarkBlockWillExit();
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchCallJSFunction: {
      EnsureSpaceForLazyDeopt();
      Register func = i.InputRegister(0);
      if (FLAG_debug_code) {
        // Check the function's context matches the context argument.
        __ cmpp(rsi, FieldOperand(func, JSFunction::kContextOffset));
        __ Assert(equal, kWrongFunctionContext);
      }
      __ Call(FieldOperand(func, JSFunction::kCodeEntryOffset));
      frame_access_state()->ClearSPDelta();
      RecordCallPosition(instr);
      break;
    }
    case kArchTailCallJSFunctionFromJSFunction:
    case kArchTailCallJSFunction: {
      Register func = i.InputRegister(0);
      if (FLAG_debug_code) {
        // Check the function's context matches the context argument.
        __ cmpp(rsi, FieldOperand(func, JSFunction::kContextOffset));
        __ Assert(equal, kWrongFunctionContext);
      }
      if (arch_opcode == kArchTailCallJSFunctionFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         i.TempRegister(0), i.TempRegister(1),
                                         i.TempRegister(2));
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
      __ PrepareCallCFunction(num_parameters);
      break;
    }
    case kArchPrepareTailCall:
      AssemblePrepareTailCall();
      break;
    case kArchCallCFunction: {
      int const num_parameters = MiscField::decode(instr->opcode());
      if (HasImmediateInput(instr, 0)) {
        ExternalReference ref = i.InputExternalReference(0);
        __ CallCFunction(ref, num_parameters);
      } else {
        Register func = i.InputRegister(0);
        __ CallCFunction(func, num_parameters);
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
      Deoptimizer::BailoutType bailout_type =
          Deoptimizer::BailoutType(MiscField::decode(instr->opcode()));
      CodeGenResult result =
          AssembleDeoptimizerCall(deopt_state_id, bailout_type);
      if (result != kSuccess) return result;
      break;
    }
    case kArchRet:
      AssembleReturn();
      break;
    case kArchStackPointer:
      __ movq(i.OutputRegister(), rsp);
      break;
    case kArchFramePointer:
      __ movq(i.OutputRegister(), rbp);
      break;
    case kArchParentFramePointer:
      if (frame_access_state()->has_frame()) {
        __ movq(i.OutputRegister(), Operand(rbp, 0));
      } else {
        __ movq(i.OutputRegister(), rbp);
      }
      break;
    case kArchTruncateDoubleToI: {
      auto result = i.OutputRegister();
      auto input = i.InputDoubleRegister(0);
      auto ool = new (zone()) OutOfLineTruncateDoubleToI(
          this, result, input, &unwinding_info_writer_);
      // We use Cvttsd2siq instead of Cvttsd2si due to performance reasons. The
      // use of Cvttsd2siq requires the movl below to avoid sign extension.
      __ Cvttsd2siq(result, input);
      __ cmpq(result, Immediate(1));
      __ j(overflow, ool->entry());
      __ bind(ool->exit());
      __ movl(result, result);
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
      __ movp(operand, value);
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
        base = rsp;
      } else {
        base = rbp;
      }
      __ leaq(i.OutputRegister(), Operand(base, offset.offset()));
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
      ASSEMBLE_IEEE754_UNOP(cos);
      break;
    case kIeee754Float64Cosh:
      ASSEMBLE_IEEE754_UNOP(cosh);
      break;
    case kIeee754Float64Exp:
      ASSEMBLE_IEEE754_UNOP(exp);
      break;
    case kIeee754Float64Expm1:
      ASSEMBLE_IEEE754_UNOP(expm1);
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
      // TODO(bmeurer): Improve integration of the stub.
      __ Movsd(xmm2, xmm0);
      MathPowStub stub(isolate(), MathPowStub::DOUBLE);
      __ CallStub(&stub);
      __ Movsd(xmm0, xmm3);
      break;
    }
    case kIeee754Float64Sin:
      ASSEMBLE_IEEE754_UNOP(sin);
      break;
    case kIeee754Float64Sinh:
      ASSEMBLE_IEEE754_UNOP(sinh);
      break;
    case kIeee754Float64Tan:
      ASSEMBLE_IEEE754_UNOP(tan);
      break;
    case kIeee754Float64Tanh:
      ASSEMBLE_IEEE754_UNOP(tanh);
      break;
    case kX64Add32:
      ASSEMBLE_BINOP(addl);
      break;
    case kX64Add:
      ASSEMBLE_BINOP(addq);
      break;
    case kX64Sub32:
      ASSEMBLE_BINOP(subl);
      break;
    case kX64Sub:
      ASSEMBLE_BINOP(subq);
      break;
    case kX64And32:
      ASSEMBLE_BINOP(andl);
      break;
    case kX64And:
      ASSEMBLE_BINOP(andq);
      break;
    case kX64Cmp8:
      ASSEMBLE_COMPARE(cmpb);
      break;
    case kX64Cmp16:
      ASSEMBLE_COMPARE(cmpw);
      break;
    case kX64Cmp32:
      ASSEMBLE_COMPARE(cmpl);
      break;
    case kX64Cmp:
      ASSEMBLE_COMPARE(cmpq);
      break;
    case kX64Test8:
      ASSEMBLE_COMPARE(testb);
      break;
    case kX64Test16:
      ASSEMBLE_COMPARE(testw);
      break;
    case kX64Test32:
      ASSEMBLE_COMPARE(testl);
      break;
    case kX64Test:
      ASSEMBLE_COMPARE(testq);
      break;
    case kX64Imul32:
      ASSEMBLE_MULT(imull);
      break;
    case kX64Imul:
      ASSEMBLE_MULT(imulq);
      break;
    case kX64ImulHigh32:
      if (instr->InputAt(1)->IsRegister()) {
        __ imull(i.InputRegister(1));
      } else {
        __ imull(i.InputOperand(1));
      }
      break;
    case kX64UmulHigh32:
      if (instr->InputAt(1)->IsRegister()) {
        __ mull(i.InputRegister(1));
      } else {
        __ mull(i.InputOperand(1));
      }
      break;
    case kX64Idiv32:
      __ cdq();
      __ idivl(i.InputRegister(1));
      break;
    case kX64Idiv:
      __ cqo();
      __ idivq(i.InputRegister(1));
      break;
    case kX64Udiv32:
      __ xorl(rdx, rdx);
      __ divl(i.InputRegister(1));
      break;
    case kX64Udiv:
      __ xorq(rdx, rdx);
      __ divq(i.InputRegister(1));
      break;
    case kX64Not:
      ASSEMBLE_UNOP(notq);
      break;
    case kX64Not32:
      ASSEMBLE_UNOP(notl);
      break;
    case kX64Neg:
      ASSEMBLE_UNOP(negq);
      break;
    case kX64Neg32:
      ASSEMBLE_UNOP(negl);
      break;
    case kX64Or32:
      ASSEMBLE_BINOP(orl);
      break;
    case kX64Or:
      ASSEMBLE_BINOP(orq);
      break;
    case kX64Xor32:
      ASSEMBLE_BINOP(xorl);
      break;
    case kX64Xor:
      ASSEMBLE_BINOP(xorq);
      break;
    case kX64Shl32:
      ASSEMBLE_SHIFT(shll, 5);
      break;
    case kX64Shl:
      ASSEMBLE_SHIFT(shlq, 6);
      break;
    case kX64Shr32:
      ASSEMBLE_SHIFT(shrl, 5);
      break;
    case kX64Shr:
      ASSEMBLE_SHIFT(shrq, 6);
      break;
    case kX64Sar32:
      ASSEMBLE_SHIFT(sarl, 5);
      break;
    case kX64Sar:
      ASSEMBLE_SHIFT(sarq, 6);
      break;
    case kX64Ror32:
      ASSEMBLE_SHIFT(rorl, 5);
      break;
    case kX64Ror:
      ASSEMBLE_SHIFT(rorq, 6);
      break;
    case kX64Lzcnt:
      if (instr->InputAt(0)->IsRegister()) {
        __ Lzcntq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Lzcntq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Lzcnt32:
      if (instr->InputAt(0)->IsRegister()) {
        __ Lzcntl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Lzcntl(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Tzcnt:
      if (instr->InputAt(0)->IsRegister()) {
        __ Tzcntq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Tzcntq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Tzcnt32:
      if (instr->InputAt(0)->IsRegister()) {
        __ Tzcntl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Tzcntl(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Popcnt:
      if (instr->InputAt(0)->IsRegister()) {
        __ Popcntq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Popcntq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Popcnt32:
      if (instr->InputAt(0)->IsRegister()) {
        __ Popcntl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Popcntl(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kSSEFloat32Cmp:
      ASSEMBLE_SSE_BINOP(Ucomiss);
      break;
    case kSSEFloat32Add:
      ASSEMBLE_SSE_BINOP(addss);
      break;
    case kSSEFloat32Sub:
      ASSEMBLE_SSE_BINOP(subss);
      break;
    case kSSEFloat32Mul:
      ASSEMBLE_SSE_BINOP(mulss);
      break;
    case kSSEFloat32Div:
      ASSEMBLE_SSE_BINOP(divss);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulss depending on the result.
      __ movaps(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kSSEFloat32Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      __ pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
      __ psrlq(kScratchDoubleReg, 33);
      __ andps(i.OutputDoubleRegister(), kScratchDoubleReg);
      break;
    }
    case kSSEFloat32Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      __ pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
      __ psllq(kScratchDoubleReg, 31);
      __ xorps(i.OutputDoubleRegister(), kScratchDoubleReg);
      break;
    }
    case kSSEFloat32Sqrt:
      ASSEMBLE_SSE_UNOP(sqrtss);
      break;
    case kSSEFloat32ToFloat64:
      ASSEMBLE_SSE_UNOP(Cvtss2sd);
      break;
    case kSSEFloat32Round: {
      CpuFeatureScope sse_scope(masm(), SSE4_1);
      RoundingMode const mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      __ Roundss(i.OutputDoubleRegister(), i.InputDoubleRegister(0), mode);
      break;
    }
    case kSSEFloat32ToInt32:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2si(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2si(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kSSEFloat32ToUint32: {
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2siq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    }
    case kSSEFloat64Cmp:
      ASSEMBLE_SSE_BINOP(Ucomisd);
      break;
    case kSSEFloat64Add:
      ASSEMBLE_SSE_BINOP(addsd);
      break;
    case kSSEFloat64Sub:
      ASSEMBLE_SSE_BINOP(subsd);
      break;
    case kSSEFloat64Mul:
      ASSEMBLE_SSE_BINOP(mulsd);
      break;
    case kSSEFloat64Div:
      ASSEMBLE_SSE_BINOP(divsd);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulsd depending on the result.
      __ Movapd(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kSSEFloat64Mod: {
      __ subq(rsp, Immediate(kDoubleSize));
      unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                       kDoubleSize);
      // Move values to st(0) and st(1).
      __ Movsd(Operand(rsp, 0), i.InputDoubleRegister(1));
      __ fld_d(Operand(rsp, 0));
      __ Movsd(Operand(rsp, 0), i.InputDoubleRegister(0));
      __ fld_d(Operand(rsp, 0));
      // Loop while fprem isn't done.
      Label mod_loop;
      __ bind(&mod_loop);
      // This instructions traps on all kinds inputs, but we are assuming the
      // floating point control word is set to ignore them all.
      __ fprem();
      // The following 2 instruction implicitly use rax.
      __ fnstsw_ax();
      if (CpuFeatures::IsSupported(SAHF)) {
        CpuFeatureScope sahf_scope(masm(), SAHF);
        __ sahf();
      } else {
        __ shrl(rax, Immediate(8));
        __ andl(rax, Immediate(0xFF));
        __ pushq(rax);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kPointerSize);
        __ popfq();
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         -kPointerSize);
      }
      __ j(parity_even, &mod_loop);
      // Move output to stack and clean up.
      __ fstp(1);
      __ fstp_d(Operand(rsp, 0));
      __ Movsd(i.OutputDoubleRegister(), Operand(rsp, 0));
      __ addq(rsp, Immediate(kDoubleSize));
      unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                       -kDoubleSize);
      break;
    }
    case kSSEFloat32Max: {
      Label compare_nan, compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          new (zone()) OutOfLineLoadFloat32NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(above, &done_compare, Label::kNear);
      __ j(below, &compare_swap, Label::kNear);
      __ Movmskps(kScratchRegister, i.InputDoubleRegister(0));
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat32Min: {
      Label compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          new (zone()) OutOfLineLoadFloat32NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(below, &done_compare, Label::kNear);
      __ j(above, &compare_swap, Label::kNear);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movmskps(kScratchRegister, i.InputDoubleRegister(1));
      } else {
        __ Movss(kScratchDoubleReg, i.InputOperand(1));
        __ Movmskps(kScratchRegister, kScratchDoubleReg);
      }
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat64Max: {
      Label compare_nan, compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          new (zone()) OutOfLineLoadFloat64NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(above, &done_compare, Label::kNear);
      __ j(below, &compare_swap, Label::kNear);
      __ Movmskpd(kScratchRegister, i.InputDoubleRegister(0));
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movsd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movsd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat64Min: {
      Label compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          new (zone()) OutOfLineLoadFloat64NaN(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(below, &done_compare, Label::kNear);
      __ j(above, &compare_swap, Label::kNear);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movmskpd(kScratchRegister, i.InputDoubleRegister(1));
      } else {
        __ Movsd(kScratchDoubleReg, i.InputOperand(1));
        __ Movmskpd(kScratchRegister, kScratchDoubleReg);
      }
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movsd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movsd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat64Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      __ pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
      __ psrlq(kScratchDoubleReg, 1);
      __ andpd(i.OutputDoubleRegister(), kScratchDoubleReg);
      break;
    }
    case kSSEFloat64Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      __ pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
      __ psllq(kScratchDoubleReg, 63);
      __ xorpd(i.OutputDoubleRegister(), kScratchDoubleReg);
      break;
    }
    case kSSEFloat64Sqrt:
      ASSEMBLE_SSE_UNOP(sqrtsd);
      break;
    case kSSEFloat64Round: {
      CpuFeatureScope sse_scope(masm(), SSE4_1);
      RoundingMode const mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      __ Roundsd(i.OutputDoubleRegister(), i.InputDoubleRegister(0), mode);
      break;
    }
    case kSSEFloat64ToFloat32:
      ASSEMBLE_SSE_UNOP(Cvtsd2ss);
      break;
    case kSSEFloat64ToInt32:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2si(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2si(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kSSEFloat64ToUint32: {
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2siq(i.OutputRegister(), i.InputOperand(0));
      }
      if (MiscField::decode(instr->opcode())) {
        __ AssertZeroExtended(i.OutputRegister());
      }
      break;
    }
    case kSSEFloat32ToInt64:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2siq(i.OutputRegister(), i.InputOperand(0));
      }
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 1);
        Label done;
        Label fail;
        __ Move(kScratchDoubleReg, static_cast<float>(INT64_MIN));
        if (instr->InputAt(0)->IsFPRegister()) {
          __ Ucomiss(kScratchDoubleReg, i.InputDoubleRegister(0));
        } else {
          __ Ucomiss(kScratchDoubleReg, i.InputOperand(0));
        }
        // If the input is NaN, then the conversion fails.
        __ j(parity_even, &fail);
        // If the input is INT64_MIN, then the conversion succeeds.
        __ j(equal, &done);
        __ cmpq(i.OutputRegister(0), Immediate(1));
        // If the conversion results in INT64_MIN, but the input was not
        // INT64_MIN, then the conversion fails.
        __ j(no_overflow, &done);
        __ bind(&fail);
        __ Set(i.OutputRegister(1), 0);
        __ bind(&done);
      }
      break;
    case kSSEFloat64ToInt64:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2siq(i.OutputRegister(0), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2siq(i.OutputRegister(0), i.InputOperand(0));
      }
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 1);
        Label done;
        Label fail;
        __ Move(kScratchDoubleReg, static_cast<double>(INT64_MIN));
        if (instr->InputAt(0)->IsFPRegister()) {
          __ Ucomisd(kScratchDoubleReg, i.InputDoubleRegister(0));
        } else {
          __ Ucomisd(kScratchDoubleReg, i.InputOperand(0));
        }
        // If the input is NaN, then the conversion fails.
        __ j(parity_even, &fail);
        // If the input is INT64_MIN, then the conversion succeeds.
        __ j(equal, &done);
        __ cmpq(i.OutputRegister(0), Immediate(1));
        // If the conversion results in INT64_MIN, but the input was not
        // INT64_MIN, then the conversion fails.
        __ j(no_overflow, &done);
        __ bind(&fail);
        __ Set(i.OutputRegister(1), 0);
        __ bind(&done);
      }
      break;
    case kSSEFloat32ToUint64: {
      Label done;
      Label success;
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 0);
      }
      // There does not exist a Float32ToUint64 instruction, so we have to use
      // the Float32ToInt64 instruction.
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2siq(i.OutputRegister(), i.InputOperand(0));
      }
      // Check if the result of the Float32ToInt64 conversion is positive, we
      // are already done.
      __ testq(i.OutputRegister(), i.OutputRegister());
      __ j(positive, &success);
      // The result of the first conversion was negative, which means that the
      // input value was not within the positive int64 range. We subtract 2^64
      // and convert it again to see if it is within the uint64 range.
      __ Move(kScratchDoubleReg, -9223372036854775808.0f);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ addss(kScratchDoubleReg, i.InputDoubleRegister(0));
      } else {
        __ addss(kScratchDoubleReg, i.InputOperand(0));
      }
      __ Cvttss2siq(i.OutputRegister(), kScratchDoubleReg);
      __ testq(i.OutputRegister(), i.OutputRegister());
      // The only possible negative value here is 0x80000000000000000, which is
      // used on x64 to indicate an integer overflow.
      __ j(negative, &done);
      // The input value is within uint64 range and the second conversion worked
      // successfully, but we still have to undo the subtraction we did
      // earlier.
      __ Set(kScratchRegister, 0x8000000000000000);
      __ orq(i.OutputRegister(), kScratchRegister);
      __ bind(&success);
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 1);
      }
      __ bind(&done);
      break;
    }
    case kSSEFloat64ToUint64: {
      Label done;
      Label success;
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 0);
      }
      // There does not exist a Float64ToUint64 instruction, so we have to use
      // the Float64ToInt64 instruction.
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2siq(i.OutputRegister(), i.InputOperand(0));
      }
      // Check if the result of the Float64ToInt64 conversion is positive, we
      // are already done.
      __ testq(i.OutputRegister(), i.OutputRegister());
      __ j(positive, &success);
      // The result of the first conversion was negative, which means that the
      // input value was not within the positive int64 range. We subtract 2^64
      // and convert it again to see if it is within the uint64 range.
      __ Move(kScratchDoubleReg, -9223372036854775808.0);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ addsd(kScratchDoubleReg, i.InputDoubleRegister(0));
      } else {
        __ addsd(kScratchDoubleReg, i.InputOperand(0));
      }
      __ Cvttsd2siq(i.OutputRegister(), kScratchDoubleReg);
      __ testq(i.OutputRegister(), i.OutputRegister());
      // The only possible negative value here is 0x80000000000000000, which is
      // used on x64 to indicate an integer overflow.
      __ j(negative, &done);
      // The input value is within uint64 range and the second conversion worked
      // successfully, but we still have to undo the subtraction we did
      // earlier.
      __ Set(kScratchRegister, 0x8000000000000000);
      __ orq(i.OutputRegister(), kScratchRegister);
      __ bind(&success);
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 1);
      }
      __ bind(&done);
      break;
    }
    case kSSEInt32ToFloat64:
      if (instr->InputAt(0)->IsRegister()) {
        __ Cvtlsi2sd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtlsi2sd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEInt32ToFloat32:
      if (instr->InputAt(0)->IsRegister()) {
        __ Cvtlsi2ss(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtlsi2ss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEInt64ToFloat32:
      if (instr->InputAt(0)->IsRegister()) {
        __ Cvtqsi2ss(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtqsi2ss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEInt64ToFloat64:
      if (instr->InputAt(0)->IsRegister()) {
        __ Cvtqsi2sd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtqsi2sd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEUint64ToFloat32:
      if (instr->InputAt(0)->IsRegister()) {
        __ movq(kScratchRegister, i.InputRegister(0));
      } else {
        __ movq(kScratchRegister, i.InputOperand(0));
      }
      __ Cvtqui2ss(i.OutputDoubleRegister(), kScratchRegister,
                   i.TempRegister(0));
      break;
    case kSSEUint64ToFloat64:
      if (instr->InputAt(0)->IsRegister()) {
        __ movq(kScratchRegister, i.InputRegister(0));
      } else {
        __ movq(kScratchRegister, i.InputOperand(0));
      }
      __ Cvtqui2sd(i.OutputDoubleRegister(), kScratchRegister,
                   i.TempRegister(0));
      break;
    case kSSEUint32ToFloat64:
      if (instr->InputAt(0)->IsRegister()) {
        __ movl(kScratchRegister, i.InputRegister(0));
      } else {
        __ movl(kScratchRegister, i.InputOperand(0));
      }
      __ Cvtqsi2sd(i.OutputDoubleRegister(), kScratchRegister);
      break;
    case kSSEUint32ToFloat32:
      if (instr->InputAt(0)->IsRegister()) {
        __ movl(kScratchRegister, i.InputRegister(0));
      } else {
        __ movl(kScratchRegister, i.InputOperand(0));
      }
      __ Cvtqsi2ss(i.OutputDoubleRegister(), kScratchRegister);
      break;
    case kSSEFloat64ExtractLowWord32:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movl(i.OutputRegister(), i.InputOperand(0));
      } else {
        __ Movd(i.OutputRegister(), i.InputDoubleRegister(0));
      }
      break;
    case kSSEFloat64ExtractHighWord32:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movl(i.OutputRegister(), i.InputOperand(0, kDoubleSize / 2));
      } else {
        __ Pextrd(i.OutputRegister(), i.InputDoubleRegister(0), 1);
      }
      break;
    case kSSEFloat64InsertLowWord32:
      if (instr->InputAt(1)->IsRegister()) {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputRegister(1), 0);
      } else {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputOperand(1), 0);
      }
      break;
    case kSSEFloat64InsertHighWord32:
      if (instr->InputAt(1)->IsRegister()) {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputRegister(1), 1);
      } else {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputOperand(1), 1);
      }
      break;
    case kSSEFloat64LoadLowWord32:
      if (instr->InputAt(0)->IsRegister()) {
        __ Movd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Movd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kAVXFloat32Cmp: {
      CpuFeatureScope avx_scope(masm(), AVX);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ vucomiss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ vucomiss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      break;
    }
    case kAVXFloat32Add:
      ASSEMBLE_AVX_BINOP(vaddss);
      break;
    case kAVXFloat32Sub:
      ASSEMBLE_AVX_BINOP(vsubss);
      break;
    case kAVXFloat32Mul:
      ASSEMBLE_AVX_BINOP(vmulss);
      break;
    case kAVXFloat32Div:
      ASSEMBLE_AVX_BINOP(vdivss);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulss depending on the result.
      __ Movaps(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kAVXFloat64Cmp: {
      CpuFeatureScope avx_scope(masm(), AVX);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ vucomisd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ vucomisd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      break;
    }
    case kAVXFloat64Add:
      ASSEMBLE_AVX_BINOP(vaddsd);
      break;
    case kAVXFloat64Sub:
      ASSEMBLE_AVX_BINOP(vsubsd);
      break;
    case kAVXFloat64Mul:
      ASSEMBLE_AVX_BINOP(vmulsd);
      break;
    case kAVXFloat64Div:
      ASSEMBLE_AVX_BINOP(vdivsd);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulsd depending on the result.
      __ Movapd(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kAVXFloat32Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(masm(), AVX);
      __ vpcmpeqd(kScratchDoubleReg, kScratchDoubleReg, kScratchDoubleReg);
      __ vpsrlq(kScratchDoubleReg, kScratchDoubleReg, 33);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vandps(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputDoubleRegister(0));
      } else {
        __ vandps(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputOperand(0));
      }
      break;
    }
    case kAVXFloat32Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(masm(), AVX);
      __ vpcmpeqd(kScratchDoubleReg, kScratchDoubleReg, kScratchDoubleReg);
      __ vpsllq(kScratchDoubleReg, kScratchDoubleReg, 31);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vxorps(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputDoubleRegister(0));
      } else {
        __ vxorps(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputOperand(0));
      }
      break;
    }
    case kAVXFloat64Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(masm(), AVX);
      __ vpcmpeqd(kScratchDoubleReg, kScratchDoubleReg, kScratchDoubleReg);
      __ vpsrlq(kScratchDoubleReg, kScratchDoubleReg, 1);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vandpd(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputDoubleRegister(0));
      } else {
        __ vandpd(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputOperand(0));
      }
      break;
    }
    case kAVXFloat64Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(masm(), AVX);
      __ vpcmpeqd(kScratchDoubleReg, kScratchDoubleReg, kScratchDoubleReg);
      __ vpsllq(kScratchDoubleReg, kScratchDoubleReg, 63);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vxorpd(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputDoubleRegister(0));
      } else {
        __ vxorpd(i.OutputDoubleRegister(), kScratchDoubleReg,
                  i.InputOperand(0));
      }
      break;
    }
    case kSSEFloat64SilenceNaN:
      __ Xorpd(kScratchDoubleReg, kScratchDoubleReg);
      __ Subsd(i.InputDoubleRegister(0), kScratchDoubleReg);
      break;
    case kX64Movsxbl:
      ASSEMBLE_MOVX(movsxbl);
      __ AssertZeroExtended(i.OutputRegister());
      break;
    case kX64Movzxbl:
      ASSEMBLE_MOVX(movzxbl);
      __ AssertZeroExtended(i.OutputRegister());
      break;
    case kX64Movsxbq:
      ASSEMBLE_MOVX(movsxbq);
      break;
    case kX64Movzxbq:
      ASSEMBLE_MOVX(movzxbq);
      __ AssertZeroExtended(i.OutputRegister());
      break;
    case kX64Movb: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ movb(operand, Immediate(i.InputInt8(index)));
      } else {
        __ movb(operand, i.InputRegister(index));
      }
      break;
    }
    case kX64Movsxwl:
      ASSEMBLE_MOVX(movsxwl);
      __ AssertZeroExtended(i.OutputRegister());
      break;
    case kX64Movzxwl:
      ASSEMBLE_MOVX(movzxwl);
      __ AssertZeroExtended(i.OutputRegister());
      break;
    case kX64Movsxwq:
      ASSEMBLE_MOVX(movsxwq);
      break;
    case kX64Movzxwq:
      ASSEMBLE_MOVX(movzxwq);
      __ AssertZeroExtended(i.OutputRegister());
      break;
    case kX64Movw: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ movw(operand, Immediate(i.InputInt16(index)));
      } else {
        __ movw(operand, i.InputRegister(index));
      }
      break;
    }
    case kX64Movl:
    case kX64TrapMovl:
      if (instr->HasOutput()) {
        if (instr->addressing_mode() == kMode_None) {
          if (instr->InputAt(0)->IsRegister()) {
            __ movl(i.OutputRegister(), i.InputRegister(0));
          } else {
            __ movl(i.OutputRegister(), i.InputOperand(0));
          }
        } else {
          Address pc = __ pc();
          __ movl(i.OutputRegister(), i.MemoryOperand());

          if (arch_opcode == kX64TrapMovl) {
            bool frame_elided = !frame_access_state()->has_frame();
            new (zone()) WasmOutOfLineTrap(this, pc, frame_elided,
                                           i.InputRegister(2), i.InputInt32(3));
          }
        }
        __ AssertZeroExtended(i.OutputRegister());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        if (HasImmediateInput(instr, index)) {
          __ movl(operand, i.InputImmediate(index));
        } else {
          __ movl(operand, i.InputRegister(index));
        }
      }
      break;
    case kX64Movsxlq:
      ASSEMBLE_MOVX(movsxlq);
      break;
    case kX64Movq:
      if (instr->HasOutput()) {
        __ movq(i.OutputRegister(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        if (HasImmediateInput(instr, index)) {
          __ movq(operand, i.InputImmediate(index));
        } else {
          __ movq(operand, i.InputRegister(index));
        }
      }
      break;
    case kX64Movss:
      if (instr->HasOutput()) {
        __ movss(i.OutputDoubleRegister(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ movss(operand, i.InputDoubleRegister(index));
      }
      break;
    case kX64Movsd:
      if (instr->HasOutput()) {
        __ Movsd(i.OutputDoubleRegister(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ Movsd(operand, i.InputDoubleRegister(index));
      }
      break;
    case kX64BitcastFI:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movl(i.OutputRegister(), i.InputOperand(0));
      } else {
        __ Movd(i.OutputRegister(), i.InputDoubleRegister(0));
      }
      break;
    case kX64BitcastDL:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movq(i.OutputRegister(), i.InputOperand(0));
      } else {
        __ Movq(i.OutputRegister(), i.InputDoubleRegister(0));
      }
      break;
    case kX64BitcastIF:
      if (instr->InputAt(0)->IsRegister()) {
        __ Movd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ movss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kX64BitcastLD:
      if (instr->InputAt(0)->IsRegister()) {
        __ Movq(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Movsd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kX64Lea32: {
      AddressingMode mode = AddressingModeField::decode(instr->opcode());
      // Shorten "leal" to "addl", "subl" or "shll" if the register allocation
      // and addressing mode just happens to work out. The "addl"/"subl" forms
      // in these cases are faster based on measurements.
      if (i.InputRegister(0).is(i.OutputRegister())) {
        if (mode == kMode_MRI) {
          int32_t constant_summand = i.InputInt32(1);
          if (constant_summand > 0) {
            __ addl(i.OutputRegister(), Immediate(constant_summand));
          } else if (constant_summand < 0) {
            __ subl(i.OutputRegister(), Immediate(-constant_summand));
          }
        } else if (mode == kMode_MR1) {
          if (i.InputRegister(1).is(i.OutputRegister())) {
            __ shll(i.OutputRegister(), Immediate(1));
          } else {
            __ leal(i.OutputRegister(), i.MemoryOperand());
          }
        } else if (mode == kMode_M2) {
          __ shll(i.OutputRegister(), Immediate(1));
        } else if (mode == kMode_M4) {
          __ shll(i.OutputRegister(), Immediate(2));
        } else if (mode == kMode_M8) {
          __ shll(i.OutputRegister(), Immediate(3));
        } else {
          __ leal(i.OutputRegister(), i.MemoryOperand());
        }
      } else {
        __ leal(i.OutputRegister(), i.MemoryOperand());
      }
      __ AssertZeroExtended(i.OutputRegister());
      break;
    }
    case kX64Lea:
      __ leaq(i.OutputRegister(), i.MemoryOperand());
      break;
    case kX64Dec32:
      __ decl(i.OutputRegister());
      break;
    case kX64Inc32:
      __ incl(i.OutputRegister());
      break;
    case kX64Push:
      if (HasImmediateInput(instr, 0)) {
        __ pushq(i.InputImmediate(0));
        frame_access_state()->IncreaseSPDelta(1);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kPointerSize);
      } else {
        if (instr->InputAt(0)->IsRegister()) {
          __ pushq(i.InputRegister(0));
          frame_access_state()->IncreaseSPDelta(1);
          unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                           kPointerSize);
        } else if (instr->InputAt(0)->IsFPRegister()) {
          // TODO(titzer): use another machine instruction?
          __ subq(rsp, Immediate(kDoubleSize));
          frame_access_state()->IncreaseSPDelta(kDoubleSize / kPointerSize);
          unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                           kDoubleSize);
          __ Movsd(Operand(rsp, 0), i.InputDoubleRegister(0));
        } else {
          __ pushq(i.InputOperand(0));
          frame_access_state()->IncreaseSPDelta(1);
          unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                           kPointerSize);
        }
      }
      break;
    case kX64Poke: {
      int const slot = MiscField::decode(instr->opcode());
      if (HasImmediateInput(instr, 0)) {
        __ movq(Operand(rsp, slot * kPointerSize), i.InputImmediate(0));
      } else {
        __ movq(Operand(rsp, slot * kPointerSize), i.InputRegister(0));
      }
      break;
    }
    case kX64Xchgb: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      __ xchgb(i.InputRegister(index), operand);
      break;
    }
    case kX64Xchgw: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      __ xchgw(i.InputRegister(index), operand);
      break;
    }
    case kX64Xchgl: {
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      __ xchgl(i.InputRegister(index), operand);
      break;
    }
    case kX64Int32x4Create: {
      CpuFeatureScope sse_scope(masm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      __ Movd(dst, i.InputRegister(0));
      __ shufps(dst, dst, 0x0);
      break;
    }
    case kX64Int32x4ExtractLane: {
      CpuFeatureScope sse_scope(masm(), SSE4_1);
      __ Pextrd(i.OutputRegister(), i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kCheckedLoadInt8:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movsxbl);
      break;
    case kCheckedLoadUint8:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movzxbl);
      break;
    case kCheckedLoadInt16:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movsxwl);
      break;
    case kCheckedLoadUint16:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movzxwl);
      break;
    case kCheckedLoadWord32:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movl);
      break;
    case kCheckedLoadWord64:
      ASSEMBLE_CHECKED_LOAD_INTEGER(movq);
      break;
    case kCheckedLoadFloat32:
      ASSEMBLE_CHECKED_LOAD_FLOAT(Movss, OutOfLineLoadFloat32NaN);
      break;
    case kCheckedLoadFloat64:
      ASSEMBLE_CHECKED_LOAD_FLOAT(Movsd, OutOfLineLoadFloat64NaN);
      break;
    case kCheckedStoreWord8:
      ASSEMBLE_CHECKED_STORE_INTEGER(movb);
      break;
    case kCheckedStoreWord16:
      ASSEMBLE_CHECKED_STORE_INTEGER(movw);
      break;
    case kCheckedStoreWord32:
      ASSEMBLE_CHECKED_STORE_INTEGER(movl);
      break;
    case kCheckedStoreWord64:
      ASSEMBLE_CHECKED_STORE_INTEGER(movq);
      break;
    case kCheckedStoreFloat32:
      ASSEMBLE_CHECKED_STORE_FLOAT(Movss);
      break;
    case kCheckedStoreFloat64:
      ASSEMBLE_CHECKED_STORE_FLOAT(Movsd);
      break;
    case kX64StackCheck:
      __ CompareRoot(rsp, Heap::kStackLimitRootIndex);
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
